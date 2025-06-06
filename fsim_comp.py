from collections import OrderedDict
from typing import Callable, Tuple, Union
import cv2
import torch
import time
from datetime import timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity
from image_similarity_measures.quality_metrics import fsim

from src.misc.utils import recur_find_ext, mkdir, log_info, set_logger
# from conicInference.engine import FileLoader


class FileLoader(torch.utils.data.Dataset):
    """A data loader.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py

    """

    def __init__(
        self,
        img_path,
        indices=None,
        preproc: Callable = None,
        resize=False,
    ):
        self.imgs = np.load(img_path, mmap_mode="r")

        self.indices = (
            indices if indices is not None else np.arange(0, self.imgs.shape[0])
        )

        self.preproc = preproc if preproc else lambda x: x
        self.resize = resize
        return

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        # RGB images
        img = np.array(self.imgs[idx]).astype("uint8")
        if self.resize or img.shape[0] != 256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = self.preproc(img)
        return idx, img

def create_loader(file_path, batch_size, num_workers):
    preproc = T.Compose(
        [
            T.ToTensor(),
            # T.Grayscale(num_output_channels=3),
            # T.Resize((256, 256)),
        ]
    )
    ds = FileLoader(file_path, preproc=preproc)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return loader


def compute_comparison(original_path, comp_path, comp_str, num_workers=2, device="cuda:0"):
    original_loader = create_loader(original_path, 1, num_workers)
    comp_loader = create_loader(comp_path, 1, num_workers)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    # fid.set_dtype(torch.float64)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(
        gaussian_kernel=False,
        data_range=1.0,
        # k2=0.4
    ).to(device)
    

    store = pd.DataFrame()
    for idx, (batch_orig, batch_comp) in tqdm(
        enumerate(zip(original_loader, comp_loader)), total=len(original_loader)
    ):
        _, batch_orig = batch_orig
        _, batch_comp = batch_comp
        # ssim_ = structural_similarity(
        #     batch_orig.squeeze().permute(1, 2, 0).cpu().numpy(),
        #     batch_comp.squeeze().permute(1, 2, 0).cpu().numpy(),
        #     data_range=1.0,
        #     channel_axis=-1,
        #     # multichannel=True,
        # )
        batch_orig, batch_comp = batch_orig.to(device), batch_comp.to(device)
        batch_orig, batch_comp = batch_orig.type(torch.float64), batch_comp.type(torch.float64)
        # fid.update(batch_orig, real=True)
        # fid.update(batch_comp, real=False)

        # psnr.update(batch_comp, batch_orig)
        ssim.update(batch_comp, batch_orig)
        ssim_ = ssim.compute().item()
        ssim.reset()
        fsim_ = fsim(
            # batch_orig.squeeze().permute(1, 2, 0).cpu().numpy(),
            # batch_comp.squeeze().permute(1, 2, 0).cpu().numpy(),
            batch_orig.squeeze().cpu().numpy(),
            batch_comp.squeeze().cpu().numpy(),
        )
        row = pd.Series(
            {
                "idx": int(idx),
                f"ssim_{comp_str}": ssim_,
                f"fsim_{comp_str}": fsim_,
            }
        )
        store = pd.concat([store, row.to_frame().T])
    return store

    fid_ = fid.compute().item()
    psnr_ = psnr.compute().item()
    ssim_ = ssim.compute().item()
    return fid_, psnr_, ssim_


if __name__ == "__main__":
    sf_path = '/mnt/romesco_cloud_workspace/atypiaTest/output_2.npy'
    sf_path2 = '/mnt/romesco_cloud_workspace/atypiaTest/output_train.npy'
    mac_path = 'output/macenko_test.npy'
    mac_path2 = 'output/macenko_cherrypick_test.npy'
    # normed_path = '/home/robj/Projects/Diffusion/outputs/debug/h_target/h_target.npy'
    nst_path = '/home/robj/Projects/Diffusion/outputs/debug2/h_target/NST_test.npy'
    nst_path_tr = '/home/robj/Projects/Diffusion/outputs/debug2/h_target/NST_train.npy'
    gt_path = 'output/hamamatsu.npy'
    ap_path = 'output/aperio.npy'
    
    mac_train = 'output/macenko_train.npy'
    gt_train = 'output/hamamatsu_train.npy'
    # fid, psnr, ssim = compute_comparison(gt_path, normed_path)
    # ssim_mac = compute_comparison(gt_path, mac_path, 'macenko')
    # ssim_mac_tr = compute_comparison(gt_train, mac_train, 'macenko')
    # ssim_sf = compute_comparison(gt_path, sf_path, 'sf')
    # ssim_sf_tr = compute_comparison(gt_train, sf_path2, 'sf')
    
    ssim_nst = compute_comparison(gt_path, nst_path, 'nst')
    ssim_nst_tr = compute_comparison(gt_train, nst_path_tr, 'nst')
    df = pd.concat([ssim_nst, ssim_nst_tr])
    df.to_csv('output/fsimCompNstDebug2.csv', index=False)
    
    # df = pd.concat([ssim_mac, ssim_mac_tr])
    # df2 = pd.concat([ssim_sf, ssim_sf_tr])
    
    # df.to_csv('output/fsimCompMacDebug2.csv', index=False)
    # df2.to_csv('output/fsimCompSFDebug2.csv', index=False)
    
    # ssim_mac2 = compute_comparison(gt_path, mac_path2, 'macenko2')
    # ssim_nst = compute_comparison(gt_path, nst_path, 'nst')
    # ssim_sf = compute_comparison(gt_path, sf_path, 'sf')

    # df = pd.merge(ssim_mac, ssim_mac2, on='idx')
    # df = pd.merge(df, ssim_nst, on='idx')
    # # df = pd.merge(ssim_mac, ssim_nst, on='idx')
    # df = pd.merge(df, ssim_sf, on='idx')
    # df.to_csv('output/stainDiffComp.csv', index=False)
    # df = pd.concat([ssim_mac, ssim_nst, ssim_sf], axis=1)
    
    # root_path = "/mnt/idg/conic/data/patches/super-resolution-resized/esrgan/"
    # share_path = "/mnt/romesco_lab_share_stainFuser/finished/conic/inference/super-resolution-resized/esrgan/resized-area/histology/he-stain-with-nuclei/hsv/diffusion/"
    # share_path_orig = "/mnt/romesco_lab_share_stainFuser/finished/conic/inference/original-resolution/histology/he-stain-with-nuclei/hsv/diffusion/"

    # original_path = f"{root_path}/resized-original/area.npy"

    # ruifrok_path = (
    #     f"{root_path}/resized-area/histology/he-stain-with-nuclei/hsv/ruifrok/"
    # )
    # vahadane_path = (
    #     f"{root_path}/resized-area/histology/he-stain-with-nuclei/hsv/vahadane/"
    # )
    # neural_path = (
    #     f"{root_path}/resized-area/histology/he-stain-with-nuclei/hsv/neural-v1/"
    # )

    # cagan_path = f"{share_path}/INFID=14/"
    # stainfuser_path = f"{share_path}/INFID=19_epoch=2_step=196608_diffusionstep=20/"

    # path_50_steps = f"{share_path}/INFID=5_epoch=2_step=196608_diffusionstep=50/"
    # path_10_steps = f"{share_path}/INFID=7_epoch=2_step=196608_diffusionstep=10/"
    # path_5_steps = f"{share_path}/INFID=8_epoch=2_step=196608_diffusionstep=5/"
    # path_100_steps = f"{share_path}/INFID=11_epoch=2_step=196608_diffusionstep=100/"
    # path_20_steps = f"{share_path}/INFID=6_epoch=2_step=196608_diffusionstep=20/"
    
    # path_256_on_256 = f"{share_path_orig}/INFID=2_epoch=2_step=196608_diffusionstep=20/"
    # path_vae_256 = f"{share_path_orig}/vae_size=256/"
    
    # path_vae_512 = f"{share_path}/vae_size=512/"

    # path_256_on_512 = f"{share_path}/INFID=23_epoch=2_step=196608_diffusionstep=20/"
    # path_512_on_256 = f"/mnt/romesco_lab_share_stainFuser/finished/conic/inference/original-resolution/histology/he-stain-with-nuclei/hsv/diffusion/INFID=24_epoch=2_step=196608_diffusionstep=20/"

    # neural_path_vae_512 = '/mnt/romesco_lab_share_stainFuser/finished/conic/inference/super-resolution-resized/esrgan/resized-area/histology/he-stain-with-nuclei/hsv/neural-v1/size=512'
    # neural_path_vae_256 = '/mnt/romesco_lab_share_stainFuser/finished/conic/inference/original-resolution/histology/he-stain-with-nuclei/hsv/neural-v1/size=256/'

    # # ruifrok_sets = recur_find_ext(ruifrok_path, [".npy"])

    # comps = {
    #     # "orig_vs_ruifrok": (original_path, ruifrok_path),
    #     # "orig_vs_vahadane": (original_path, vahadane_path),
    #     # "orig_vs_neural": (original_path, neural_path),
    #     # "orig_vs_cagan": (original_path, cagan_path),
    #     # "orig_vs_stainfuser": (original_path, stainfuser_path),
    #     # "orig_vs_256_on_256": (original_path, path_256_on_256),
    #     # "orig_vs_256_on_512": (original_path, path_256_on_512),
    #     # "orig_vs_512_on_256": (original_path, path_512_on_256),
    #     # "orig_vs_vae_256": (original_path, path_vae_256),
    #     # "orig_vs_vae_512": (original_path, path_vae_512),
    #     # "orig_vs_orig": (original_path, original_path),
    #     # "orig_vs_50_steps": (original_path, path_50_steps),
    #     # "orig_vs_10_steps": (original_path, path_10_steps),
    #     # "orig_vs_5_steps": (original_path, path_5_steps),
    #     # "orig_vs_100_steps": (original_path, path_100_steps),
    #     # "orig_vs_20_steps": (original_path, path_20_steps),
    #     "orig_vs_neural_vae_256": (original_path, neural_path_vae_256),
    #     "orig_vs_neural_vae_512": (original_path, neural_path_vae_512),
    # }

    # OUT_PATH = "outputs/conicInf/ablations/imageQuality/"
    # mkdir(OUT_PATH)

    # LOG_DIR = "outputs/conicInf/ablations/logs/"
    # mkdir(LOG_DIR)
    # set_logger(f"{LOG_DIR}/imQualityMetrics.log")
    
    # out_file_name = 'image_quality_neural_raw.csv'

    # df = pd.DataFrame()
    # for comp_name, comp_path_set in comps.items():
    #     comp_paths = recur_find_ext(comp_path_set[1], [".npy"])
    #     log_info(f"Computing {comp_name}, {len(comp_paths)} sets to do")
    #     start_overall = time.perf_counter()
    #     if len(comp_paths) > 1:
    #         for idx, comp_path in enumerate(comp_paths):
    #             start = time.perf_counter()
    #             fid, psnr, ssim = compute_comparison(comp_path_set[0], comp_path)
    #             image_code = Path(comp_path).stem
    #             result = pd.Series(
    #                 {
    #                     "comparsion": comp_name,
    #                     "imageCode": Path(comp_path).stem,
    #                     "FID": fid,
    #                     "PSNR": psnr,
    #                     "SSIM": ssim,
    #                 }
    #             )
    #             df = pd.concat([df, result.to_frame().T])
    #             df.to_csv(f"{OUT_PATH}/{out_file_name}", index=False)
    #             end = time.perf_counter()
    #             log_info(
    #                 f"Finished {comp_name} {idx+1}/{len(comp_paths)} in {timedelta(seconds=end-start)}"
    #             )
    #     else:
    #         start = time.perf_counter()
    #         fid, psnr, ssim = compute_comparison(comp_path_set[0], comp_paths[0])
    #         result = pd.Series(
    #             {
    #                 "comparsion": comp_name,
    #                 "imageCode": "n/a",
    #                 "FID": fid,
    #                 "PSNR": psnr,
    #                 "SSIM": ssim,
    #             }
    #         )
    #         df = pd.concat([df, result.to_frame().T])
    #         df.to_csv(f"{OUT_PATH}/{out_file_name}", index=False)
    #         end = time.perf_counter()
    #         log_info(f"Finished {comp_name} in {timedelta(seconds=end-start)}")
    #     end_overall = time.perf_counter()
    #     log_info(
    #         f"Finished {comp_name} in {timedelta(seconds=end_overall-start_overall)}"
    #     )

    # # compute_comparison(original_path, ruifrok_sets[0])