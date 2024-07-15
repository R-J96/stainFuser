import torch
import random
import itertools
import pandas as pd
import numpy as np
import os

from pathlib import Path
from collections import OrderedDict
from typing import Optional, Union, Dict, Tuple
import torchvision.transforms as T

from src.misc.utils import (
    imread,
    recur_find_ext,
    create_image_transform,
    inverse_transform,
)


class PatchInferenceDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for patch-based inference with the StainFuser model.

    Args:
        source_path (Union[str, Path]): Path to the source image(s). Can be a directory containing PNG files or a single NumPy file.
        target_path (Union[str, Path]): Path to the target image.
        return_numpy (bool, optional): Whether to return images as NumPy arrays. If False, images will be transformed to tensors. If True, raw NumPy arrays will be returned. Defaults to False.

    Attributes:
        source_files (Union[List[str], np.memmap]): List of source file paths or a memory-mapped NumPy array.
        target (np.ndarray): The target image as a NumPy array.
        target_size (int): The size of the target image.
        transform (Compose): Composed image transformations to be applied to the input images.
        target_t (torch.Tensor): The transformed target image tensor.
    """

    def __init__(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path],
        return_numpy: bool = False,
    ):
        if os.path.isdir(source_path):
            self.source_files = recur_find_ext(source_path, [".png"])
            source_size = imread(self.source_files[0]).shape[0]
        elif os.path.isfile(source_path):
            if str(source_path).endswith(".npy"):
                self.source_files = np.load(source_path, mmap_mode="r")
            else:
                raise ValueError("Unsupported source file type")
            source_size = self.source_files.shape[0]

        self.return_numpy = return_numpy
        self.target = imread(target_path)
        self.target_size = self.target.shape[0]

        self.transform = [T.ToTensor()]

        if source_size != self.target.shape[0]:
            self.transform.append(T.Resize(self.target_size, antialias=True))
        self.transform = T.Compose(self.transform)

        self.target_t = self.transform(self.target)

    def __len__(self) -> int:
        if type(self.source_files) == list:
            return len(self.source_files)
        elif type(self.source_files) == np.memmap:
            return self.source_files.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        if type(self.source_files) == list:
            source = imread(self.source_files[idx])
        elif type(self.source_files) == np.memmap:
            source = self.source_files[idx]
        source = self.transform(source.copy())
        if type(self.source_files) == list:
            source_path = Path(self.source_files[idx]).stem
        # if np array return the index
        elif type(self.source_files) == np.memmap:
            source_path = f"{idx:0>4}"
        return source, source_path


class TrainDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for preprocessing and loading data for pretraining.

    Args:
        source_dir (Union[str, Path]): Path to source images.
        target_dir (Union[str, Path]): Path to target images.
        stylized_dir (Union[str, Path]): Path to stylized image sets.
        input_resolution (str, optional): Resolution of input images. Defaults to "40x".
        input_size (int, optional): Size of input images. Defaults to 1024.
        output_resolution (str, optional): Resolution of output images. Defaults to "40x".
        output_size (int, optional): Size of output images. Defaults to 512.
        style_subset (Optional[Union[list[str], int]], optional): Subset of style sets to consider.
            Can be an integer indicating the number of random styles to select or a path to a txt file with a list of style sets.
            Defaults to None.
        return_numpy (bool, optional): Whether to return images as NumPy arrays.
            If False, images will be transformed to tensors. If True, raw NumPy arrays will be returned.
            Defaults to False.
        mixed_resolution (bool, optional): Whether to consider mixed resolutions for image transformations. If True, ignores output_resolution. Defaults to False.
    """

    def __init__(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        stylized_dir: Union[str, Path],
        input_resolution: str = "40x",
        input_size: int = 1024,
        output_resolution: str = "40x",
        output_size: int = 512,
        style_subset: Optional[Union[list[str], int]] = None,
        return_numpy: bool = False,
        mixed_resolution: bool = False,  # ignore output_resolution
    ) -> None:
        self.return_numpy = return_numpy
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.stylized_dir = stylized_dir
        self.input_resolution = input_resolution
        self.input_size = input_size
        self.output_resolution = output_resolution
        self.output_size = output_size
        self.mixed_resolution = mixed_resolution

        self.source_paths = recur_find_ext(self.source_dir, [".png"])
        source_stems = [Path(x).stem for x in self.source_paths]
        self.source_lookup = OrderedDict(
            (Path(k).stem, v) for k, v in zip(self.source_paths, self.source_paths)
        )

        self.target_paths = recur_find_ext(self.target_dir, [".png"])
        self.target_lookup = {
            Path(k).stem: v for k, v in zip(self.target_paths, self.target_paths)
        }

        self.style_set_paths = recur_find_ext(self.stylized_dir, [".npy"])

        # filter down to just those with target set done
        target_stems = [Path(x).stem for x in self.style_set_paths]

        if isinstance(style_subset, list):
            stem_df = pd.read_csv(style_subset, header=None, names=["ID"])
            target_stems = [x for x in target_stems if x in stem_df["ID"].tolist()]
        elif isinstance(style_subset, int):
            assert style_subset <= len(target_stems)
            random.seed(42)
            target_stems = random.sample(target_stems, style_subset)
        elif isinstance(style_subset, str):
            if not os.path.exists(style_subset):
                raise ValueError("Style path" f" `{style_subset}` does not exist!")
            subset_file = open(style_subset)
            subset_string = subset_file.read().strip()
            target_stems = subset_string.split("\n")
            subset_file.close()
        else:
            raise ValueError("Unknown style subset type" f" `{type(style_subset)}`")

        self.paired_stems = list(itertools.product(source_stems, target_stems))

        self.mmap = {
            target_id: np.load(f"{self.stylized_dir}/{target_id}.npy", mmap_mode="r")
            for target_id in target_stems
        }

        if self.mixed_resolution:
            self.transform_src_trg = {}
            self.transform_tsf = {}

            self.transform_src_trg["40x"] = create_image_transform(
                input_resolution=self.input_resolution,
                output_size=self.output_size,
                output_resolution="40x",
                is_tsf=False,
            )
            self.transform_src_trg["20x"] = create_image_transform(
                input_resolution=self.input_resolution,
                output_size=self.output_size,
                output_resolution="20x",
                is_tsf=False,
            )

            self.transform_tsf["40x"] = create_image_transform(
                input_resolution=self.input_resolution,
                output_size=self.output_size,
                output_resolution="40x",
                is_tsf=True,
            )
            self.transform_tsf["20x"] = create_image_transform(
                input_resolution=self.input_resolution,
                output_size=self.output_size,
                output_resolution="20x",
                is_tsf=True,
            )

        else:
            # TODO: experiment with kornia gpu bound transforms for speedup?
            self.transform_src_trg = create_image_transform(
                input_resolution=self.input_resolution,
                output_size=self.output_size,
                output_resolution=self.output_resolution,
                is_tsf=False,
            )

            self.transform_tsf = create_image_transform(
                input_resolution=self.input_resolution,
                output_size=self.output_size,
                output_resolution=self.output_resolution,
                is_tsf=True,
            )

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.paired_stems)

    def __getitem__(
        self, idx
    ) -> Union[Dict[str, torch.FloatTensor], Dict[str, np.ndarray]]:
        """
        Get item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Union[Dict[str, torch.FloatTensor], Dict[str, np.ndarray]]: Dictionary containing source images, target images and transformed images
        """
        idx_pair = self.paired_stems[idx]
        source_path = self.source_lookup[idx_pair[0]]
        target_path = self.target_lookup[idx_pair[1]]
        source_idx = list(self.source_lookup.keys()).index(idx_pair[0])

        source_img_n = imread(source_path)
        target_img_n = imread(target_path)
        stylized_img_n = self.mmap[idx_pair[1]][source_idx]

        if not self.return_numpy:
            if self.mixed_resolution:
                random_choice = random.randint(0, 1)
                if random_choice == 0:
                    transform_src_trg = self.transform_src_trg["20x"]
                    transform_tsf = self.transform_tsf["20x"]
                else:
                    transform_src_trg = self.transform_src_trg["40x"]
                    transform_tsf = self.transform_tsf["40x"]
            else:
                transform_src_trg = self.transform_src_trg
                transform_tsf = self.transform_tsf

            source_img = transform_src_trg(np.array(source_img_n))
            target_img = transform_src_trg(np.array(target_img_n))
            stylized_img = transform_tsf(np.array(stylized_img_n))

            return {
                "Source Images": source_img,
                "Target Images": target_img,
                "Transformed Images": stylized_img,
            }
        else:
            return {
                "Source Images": inverse_transform(source_img, is_tsf=False),
                "Target Images": inverse_transform(target_img, is_tsf=False),
                "Transformed Images": inverse_transform(stylized_img, is_tsf=True),
            }
