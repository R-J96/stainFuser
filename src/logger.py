import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import matplotlib.pyplot as plt

from src.misc.utils import imwrite


class ImageLogger(Callback):
    def __init__(
        self,
        dirpath=None,
        batch_frequency=2000,
        max_images=4,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        save_separate=False,
        dpi=300,
        save_recon=False,
        diffusion_steps=50,
        save_latent=False,
        log_images_kwargs=None,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if dirpath is None:
            self.dirpath = f"{os.getcwd()}/images_debug/"
        else:
            self.dirpath = dirpath
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.save_separate = save_separate
        self.dpi = dpi
        self.save_recon = save_recon
        self.diffusion_steps = diffusion_steps
        self.save_latent = save_latent

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, split)
        grid_dict = {}
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if k.split("_")[0] not in ["src", "tgt"]:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            # filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                global_step, current_epoch, batch_idx, k
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            if self.save_separate:
                imwrite(path, grid)
                # Image.fromarray(grid).save(path)
            grid_dict[k] = grid
        final_image = self.concatenate_images(
            image_dict=grid_dict, axis=0, temp_output_dir=root
        )
        filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
            global_step, current_epoch, batch_idx, "stack"
        )
        imwrite(os.path.join(root, filename), final_image)
        # Image.fromarray(final_image).save(os.path.join(root, filename))

    def concatenate_images(self, image_dict, axis=1, temp_output_dir=""):
        image_list = []
        if self.save_recon:
            sequence = ["tgt_recon", "src_recon", "tsf_recon", "tsf_pred"]
        else:
            sequence = ["tgt_ori", "src_ori", "tsf_ori", "tsf_pred"]
        for item in sequence:
            # image_list.append(image_dict[item])
            image_list.append(self.draw_text(image_dict[item], item, temp_output_dir))
        # Concatenate the images along the specified axis
        concatenated_image = np.concatenate(image_list, axis=axis)
        return concatenated_image

    def draw_text(self, image, text, temp_output_dir):
        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor("black")

        ax.imshow(image)
        ax.axis("off")
        # Add text on the left of the first subplot
        ax.text(
            -0.01,
            0.5,
            text.upper(),
            rotation=90,
            transform=ax.transAxes,
            color="white",
            va="center",
            ha="right",
            fontsize=12,
        )
        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.02)
        temp_save_name = f"{temp_output_dir}/{text}.png"
        plt.savefig(temp_save_name, bbox_inches="tight", pad_inches=0, dpi=self.dpi)
        image = Image.open(temp_save_name)
        os.remove(temp_save_name)
        return image

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and  # batch_idx % self.batch_freq == 0
            # hasattr(pl_module, "log_images") and
            # callable(pl_module.log_images) and
            self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images, noise_latent = self.log_images_from_controlNet(
                    batch=batch, pl_module=pl_module, **self.log_images_kwargs
                )

            if self.save_latent:
                noise_latent = noise_latent.detach().cpu().numpy()
                root = os.path.join(pl_module.logger.save_dir, "image_log", split)
                filename = "gs-{:06}_e-{:06}_b-{:06}_{}".format(
                    pl_module.global_step, pl_module.current_epoch, batch_idx, "latent"
                )
                os.makedirs(root, exist_ok=True)
                np.save(os.path.join(root, filename), noise_latent)

            for k in images:
                images[k] = images[k].float()
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                self.dirpath,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            if is_train:
                pl_module.train()

    def get_ori_input(self, batch):
        src_ori = batch["Source Images"].type(torch.float32)
        tgt_ori = batch["Target Images"].type(torch.float32)
        tsf_ori = batch["Transformed Images"].type(torch.float32)

        return src_ori, tgt_ori, tsf_ori

    @torch.no_grad()
    def log_images_from_controlNet(self, batch, pl_module, N=4, n_row=2, **kwarg):
        log = dict()

        if not self.save_recon:
            src_ori, tgt_ori, tsf_ori = self.get_ori_input(batch=batch)
            log["tsf_ori"] = tsf_ori
            log["src_ori"] = src_ori
            log["tgt_ori"] = tgt_ori
        else:
            src_latents, tgt_latents, tsf_latents = pl_module.get_input(batch=batch)
            log["tsf_recon"] = pl_module.decode_first_stage(
                tsf_latents
            )  # torch.Size([4, 3, 512, 512])
            log["src_recon"] = src_latents
            log["tgt_recon"] = pl_module.decode_first_stage(tgt_latents)

        # log["tsf_pred"], noise_latent = pl_module.denoising(batch=batch, diffusion_steps=self.diffusion_steps)

        pred, noise_latent = pl_module.denoising(
            batch=batch, diffusion_steps=self.diffusion_steps
        )

        log["tsf_pred"] = pred
        # log["tsf_predT"] = torch.tanh(pred)

        return log, noise_latent

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
