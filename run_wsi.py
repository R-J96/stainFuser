from typing import Union, List, Dict
import os
import time
import torch
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import timedelta

from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from src.misc.model_utils import load_stainFuser
from src.models.wsi import WSIStainFuser
from src.misc.utils import (
    mkdir,
    recur_find_ext,
    npy2pyramid,
    rmdir,
    create_target_inf,
    set_logger,
    log_info,
)


def run_wsi_inference(
    wsi_path: str,
    msk_path: Union[str, None],
    cache_dir: str,
    model: WSIStainFuser,
    batch_size: int,
    num_workers: int,
    inference_resolution: Dict[str, float] = {"units": "mpp", "resolution": 0.5},
    on_gpu: bool = True,
) -> None:
    """
    Runs whole slide image (WSI) inference with stainFuser.

    Args:
        wsi_path (str): Path to the whole slide image file.
        msk_path (Union[str, None]): Path to the mask file (if available), or None.
        cache_dir (str): Path to the directory where the raw output will be cached.
        model (WSIStainFuser): The model instance to be used for inference.
        batch_size (int): Batch size for the inference process.
        num_workers (int): Number of worker processes to use for data loading and post-processing.
        inference_resolution (Dict[str, float], optional): Dictionary containing the resolution units and resolution value for inference. Defaults to {"units": "mpp", "resolution": 0.5}.
        on_gpu (bool, optional): Whether to perform inference on GPU. Defaults to True.

    Returns:
        None
    """
    segmentor = SemanticSegmentor(
        model=model,
        num_loader_workers=num_workers,
        num_postproc_workers=num_workers,
        batch_size=batch_size,
    )

    ioconfig = IOSegmentorConfig(
        input_resolutions=[inference_resolution],
        output_resolutions=[inference_resolution] * 3,  # ! for the 3 channels (RGB)
        save_resolution=inference_resolution,
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
    )

    segmentor.predict(
        imgs=[wsi_path],
        masks=[msk_path] if type(msk_path) == str else None,
        mode="wsi",
        on_gpu=on_gpu,
        ioconfig=ioconfig,
        crash_on_exception=True,
        save_dir=f"{cache_dir}/raw/",
    )
    return


def run_post_processing(
    cache_dir: str,
    output_dir: str,
    wsi_name: str,
    resolution: float = 0.5,
    background_color: List[int] = [255, 255, 255],
) -> None:
    """
    Performs post-processing on whole slide image data.

    Args:
        cache_dir (str): Path to the directory containing the cached raw data.
        output_dir (str): Path to the directory where the processed image will be saved.
        wsi_name (str): Name of the whole slide image.
        resolution (float, optional): Resolution of the output image. Defaults to 0.5mpp.
        background_color (List[int], optional): RGB values for the background color. Defaults to [255, 255, 255] (white).

    Returns:
        None
    """
    r, g, b = (
        np.load(f"{cache_dir}/raw/0.raw.0.npy", mmap_mode="r"),
        np.load(f"{cache_dir}/raw/0.raw.1.npy", mmap_mode="r"),
        np.load(f"{cache_dir}/raw/0.raw.2.npy", mmap_mode="r"),
    )
    r_, g_, b_ = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)
    img = np.dstack((r_, g_, b_))

    # ! raw results have a black background for non-tissue/masked regions
    # this will convert the black background to whatever background color is specified
    zero_indices = np.sum(img, axis=2) == 0
    img[zero_indices] = np.array(background_color, dtype=np.uint8)

    # save the image
    npy2pyramid(
        save_path=f"{output_dir}/{wsi_name}.tiff",
        image=img,
        resolution=resolution,
    )

    # sleep for 1 second to prevent file locking, can adjust this as needed
    time.sleep(1)

    # clean up cache
    rmdir(f"{cache_dir}/raw/")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", default=8, type=int)
    # parser.add_argument("--ckpt_path", default="checkpoints/checkpoint.safetensors", type=str)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", default="src/configs", type=str)
    # parser.add_argument("--target_path", default="data/targets/3021.png", type=str)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--output_dir", default="output/", type=str)
    # parser.add_argument("--wsi_dir", type=str, default="data/wsis/")
    parser.add_argument("--wsi_dir", type=str, required=True)
    # parser.add_argument("--msk_dir", type=str, default="data/masks/")
    parser.add_argument("--msk_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="cache/")
    parser.add_argument(
        "--file_list",
        type=str,
        required=False,
        help="Optional path to list of file stems in csv format. If provided, only WSIs in the list will be processed.",
    )
    parser.add_argument("--log_path", default="logs/", type=str)
    parser.add_argument("--diffusion_step", default=20, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use fp16 inference. Default is False.",
    )
    args = parser.parse_args()

    CACHE_DIR = args.cache_dir
    OUT_DIR = args.output_dir
    LOG_DIR = args.log_path

    mkdir(CACHE_DIR)
    mkdir(OUT_DIR)
    mkdir(LOG_DIR)

    set_logger(f"{LOG_DIR}/log-inference.log")
    log_info(args)

    target = create_target_inf(args.target_path)

    wsi_paths = recur_find_ext(args.wsi_dir, [".svs", ".ndpi", ".tiff", ".jp2"])

    if args.file_list is not None:
        file_list = pd.read_csv(args.file_list)
        wsi_stems = file_list["wsi_stem"].tolist()
        wsi_paths = [x for x in wsi_paths if Path(x).stem in wsi_stems]
        assert (
            len(wsi_paths) > 0
        ), "No WSIs found in file list found in WSI directory. Exiting."

    # initialise model
    log_info("Loading model")
    start = time.perf_counter()
    model = load_stainFuser(pretrained=args.ckpt_path, config_dir=args.config_path)

    model = WSIStainFuser(
        model,
        target=target,
        diffusion_steps=args.diffusion_step,
        fp16=args.fp16,
    )
    model.requires_grad_(False)
    end = time.perf_counter()
    log_info(f"Model loaded in {timedelta(seconds=end-start)}")

    log_info(f"Running inference on {len(wsi_paths)} WSIs")
    for wsi_path in wsi_paths:
        wsi_name = Path(wsi_path).stem
        log_info(f"Running inference on {wsi_name}")

        # run inference
        start = time.perf_counter()
        run_wsi_inference(
            wsi_path=wsi_path,
            msk_path=(
                f"{args.msk_dir}/{wsi_name}.png"
                if os.path.exists(f"{args.msk_dir}/{wsi_name}.png")
                else None
            ),
            cache_dir=CACHE_DIR,
            model=model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            on_gpu=True if torch.cuda.is_available() else False,
        )
        end = time.perf_counter()
        log_info(f"Inference on {wsi_name} completed in {timedelta(seconds=end-start)}")

        # create new WSI
        start = time.perf_counter()
        run_post_processing(
            cache_dir=CACHE_DIR,
            output_dir=OUT_DIR,
            wsi_name=wsi_name,
            resolution=0.5,
        )
        end = time.perf_counter()
        log_info(
            f"Post-processing on {wsi_name} completed in {timedelta(seconds=end-start)}"
        )
    log_info("Finished exiting")
