import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from src.models.arch import StainFuserArchitecture
from src.recipe import StainFuserEngine
from src.dataset import PatchInferenceDataset
from src.misc.utils import (
    inverse_transform,
    mkdir,
    save_image,
    dispatch_processing,
)
from src.misc.model_utils import load_stainFuser


@torch.no_grad()
def run_patch_inference(
    engine: StainFuserArchitecture,
    dataset: PatchInferenceDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """
    Run patch-based inference on the given dataset using the specified engine and device.

    Args:
        engine (StainFuserArchitecture): The StainFuser model to use for inference.
        dataset (PatchInferenceDataset): The dataset to run inference on.
        device (torch.device): The device to run inference on (e.g., CPU or GPU).
        args (InferenceArgs): Arguments for the inference process.

    Returns:
        None
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    )

    batch = {
        "Target Images": dataset.target_t.unsqueeze(0)
        .repeat(args.batch_size, 1, 1, 1)
        .to(device)
    }

    if args.save_fmt == "npy":
        output_store = np.zeros(
            (len(loader.dataset), dataset.target_size, dataset.target_size, 3),
            dtype=np.uint8,
        )

    # iterate over the dataset
    last_idx = len(loader) - 1
    for batch_idx, dataitem in tqdm(enumerate(loader), total=len(loader)):
        last_batch = batch_idx == last_idx
        source_t, source_stems = dataitem
        if last_batch:
            batch = {
                "Target Images": dataset.target_t.unsqueeze(0)
                .repeat(source_t.shape[0], 1, 1, 1)
                .to(device)
            }
        batch["Source Images"] = source_t.to(device)

        with torch.no_grad():
            if args.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output, _ = engine.denoising_inference(
                        batch=batch, diffusion_steps=args.diffusion_steps
                    )
            else:
                output, _ = engine.denoising_inference(
                    batch=batch, diffusion_steps=args.diffusion_steps
                )
            if device.type == "cuda":
                torch.cuda.synchronize()

        # convert back to numpy
        output = inverse_transform(output, True, True)

        # save images as needed
        if args.save_fmt == "npy":
            if last_batch:
                last_batch_size = source_t.shape[0]
                output_store[-last_batch_size:] = output
            else:
                output_store[
                    batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
                ] = output
        else:
            run_list = []
            for idx, (img, stem) in enumerate(zip(output, source_stems)):
                run_list.append([save_image, img, args.output_dir, stem, args.save_fmt])
            _ = dispatch_processing(run_list, num_workers=2, crash_on_exception=True)

    if args.save_fmt == "npy":
        np.save(os.path.join(args.output_dir, "output.npy"), output_store)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--ckpt_path", default="checkpoints/checkpoint.pth", type=str)
    # parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", default="src/configs", type=str)
    # parser.add_argument("--source_path", default="data/sources/", type=str)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--target_path", default="output/h_target.png", type=str)
    # parser.add_argument("--target_path", type=str, required=True)
    # parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--diffusion_steps", default=20, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_fmt", default="npy", choices=["npy", "png"])
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use fp16 inference. Default is False.",
    )
    args = parser.parse_args()

    if "." in args.save_fmt:
        args.save_fmt = args.replace(".", "")

    mkdir(args.output_dir)

    dataset = PatchInferenceDataset(
        source_path=args.source_path,
        target_path=args.target_path,
        return_numpy=True if args.save_fmt == 'npy' else False
    )

    model = load_stainFuser(pretrained=args.ckpt_path, config_dir="src/configs")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    engine = StainFuserEngine(arch=model, optim_configs=None)
    engine.to(device)

    run_patch_inference(engine, dataset, device, args)
