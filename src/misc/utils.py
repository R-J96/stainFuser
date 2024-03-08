import re
import collections
import copy
import inspect
import itertools
import logging
import operator
import os
import pathlib
import shutil
import sys
import zipfile
import torch
import torch.multiprocessing as mp
import torchvision.transforms as T
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from typing import Union

import cv2
import numpy as np
import ujson as json
import yaml
from PIL import ImageColor
from scipy import ndimage
from termcolor import colored
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

PERCEPTIVE_COLORS = [
    # "#000000", # ! dont use black
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
    "#6A3A4C",
    "#83AB58",
    "#001C1E",
    "#D1F7CE",
    "#004B28",
    "#C8D0F6",
    "#A3A489",
    "#806C66",
    "#222800",
    "#BF5650",
    "#E83000",
    "#66796D",
    "#DA007C",
    "#FF1A59",
    "#8ADBB4",
    "#1E0200",
    "#5B4E51",
    "#C895C5",
    "#320033",
    "#FF6832",
    "#66E1D3",
    "#CFCDAC",
    "#D0AC94",
    "#7ED379",
    "#012C58",
]
PERCEPTIVE_COLORS = np.array(PERCEPTIVE_COLORS)[3:]
PERCEPTIVE_COLORS_RGB = [ImageColor.getcolor(v, "RGB") for v in PERCEPTIVE_COLORS]


def align_height_width(a: np.ndarray, b: np.ndarray):
    """Aligning the height and width of two images.

    Images are assumed to be in of shape (H,W,...)
    """
    assert len(a.shape) == len(b.shape)
    assert all(va == vb for va, vb in list(zip(a.shape, b.shape))[2:])

    new_shape = list(a.shape)
    new_shape[0] = max(a.shape[0], b.shape[0])
    new_shape[1] = max(a.shape[1], b.shape[1])

    new_a = np.zeros(new_shape, dtype=a.dtype)
    new_a[: a.shape[0], : a.shape[1]] = a
    new_b = np.zeros(new_shape, dtype=b.dtype)
    new_b[: b.shape[0], : b.shape[1]] = b
    return new_a, new_b


def flatten_list(a_list):
    """Flatten a nested list."""
    return list(itertools.chain(*a_list))


def imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def imwrite(path, img):
    return cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def channel_masks(arr, channel_vals: Union[list, np.ndarray]):
    """Assume last channel."""
    sel = np.full(arr.shape[:-1], True, dtype=bool)
    for idx, val in enumerate(channel_vals):
        sel &= arr[..., idx] == val
    return sel


def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


def get_bounding_box(img, box=False):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    if box:  # top-left, bot-right in XY form
        return np.array([cmin, rmin, cmax, rmax])
    return np.array([rmin, rmax, cmin, cmax])


def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array

    Returns:
        x: cropped array

    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def mkdir(dir_path: str):
    """Make directory.

    Args:
        dir_path (str): If `None` is provided, nothing happens.

    """
    if dir_path is None:
        return
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def recur_find_ext(root_dir, ext_list, followlinks=True):
    """Recursively find all files in directories end with the `ext` such as `ext='.png'`.
    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.
    Returns:
        file_path_list (list): sorted list of filepaths.
    """
    # turn "." into a literal character in regex
    patterns = [v.replace(".", "\.") for v in ext_list]
    patterns = [f".*{v}$" for v in patterns]

    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir, followlinks=followlinks):
        for file_name in file_list:
            has_ext_flags = [
                re.match(pattern, file_name) is not None for pattern in patterns
            ]
            if any(has_ext_flags):
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def get_inst_centroid(inst_map):
    """Get instance centroids given an input instance map.

    Args:
        inst_map: input instance map

    Returns:
        array of centroids

    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


def center_pad_to_shape(img, size, cval=255):
    """Pad input image."""
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


def color_deconvolution(rgb, stain_mat):
    """Apply colour deconvolution."""
    log255 = np.log(255)  # to base 10, not base e
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float + 1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    output = np.floor(output + 0.5).astype("uint8")
    return output


def setup_logger(log_path):
    logging.basicConfig(
        level=logging.DEBUG,
        format="|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def log_debug(msg):
    (
        frame,
        filename,
        line_number,
        function_name,
        lines,
        index,
    ) = inspect.getouterframes(inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.debug("{i} {m}".format(i="." * indentation_level, m=msg))


def log_info(msg):
    (
        frame,
        filename,
        line_number,
        function_name,
        lines,
        index,
    ) = inspect.getouterframes(inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info("{i} {m}".format(i="." * indentation_level, m=msg))


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects,
    but the warning is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def wrap_func(idx, func, *args):
    """A wrapper so that any functions can be run
    with `dispatch_processing`.
    """
    try:
        return idx, func(*args)
    except Exception as exception_obj:
        # cache the exception stack trace
        # so that we can print out later if need
        print(exception_obj)
        exception_info = sys.exc_info()
        return [exception_obj, exception_info], idx, None


def worker_func(run_idx, func, crash_on_exception, *args):
    result = func(*args)
    if len(result) == 3 and crash_on_exception:
        raise result[0]
    elif len(result) == 3:
        result = result[1:]
    return run_idx, result


def multiproc_dispatcher_torch(
    data_list, num_workers=0, show_pbar=True, crash_on_exception=False
):
    """
    data_list is a list of [[func, arg1, arg2, etc.]]
    Results are always sorted according to source position
    """
    result_list = []

    if show_pbar:
        pbar = tqdm(total=len(data_list), ascii=True, position=0)

    if num_workers > 0:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(num_workers)
        for run_idx, dat in enumerate(data_list):
            func = dat[0]
            args = dat[1:]
            result = pool.apply_async(
                worker_func, (run_idx, func, crash_on_exception, *args)
            )
            result_list.append(result)
            if show_pbar:
                pbar.update()
        pool.close()
        pool.join()
        result_list = [result.get() for result in result_list]
    else:
        for run_idx, dat in enumerate(data_list):
            func = dat[0]
            args = dat[1:]
            result = worker_func(run_idx, func, crash_on_exception, *args)
            result_list.append(result)
            if show_pbar:
                pbar.update()

    if show_pbar:
        pbar.close()

    result_list = sorted(result_list, key=lambda k: k[0])
    result_list = [v[1] for v in result_list]
    return result_list


def dispatch_processing(
    data_list, num_workers=0, show_progress=True, crash_on_exception=False
):
    """
    data_list is alist of [[func, arg1, arg2, etc.]]
    Resutls are alway sorted according to source position
    """

    def handle_wrapper_results(result):
        if len(result) == 3 and crash_on_exception:
            exception_obj, exception_info = result[0]
            logging.info(exception_obj)
            del exception_info
            raise exception_obj
        elif len(result) == 3:
            result = result[1:]
        return result

    executor = None if num_workers <= 1 else ProcessPoolExecutor(num_workers)

    result_list = []
    future_list = []

    progress_bar = tqdm(
        total=len(data_list), ascii=True, position=0, disable=not show_progress
    )
    with logging_redirect_tqdm([logging.getLogger()]):
        for run_idx, dat in enumerate(data_list):
            func = dat[0]
            args = dat[1:]
            if num_workers > 1:
                future = executor.submit(wrap_func, run_idx, func, *args)
                future_list.append(future)
            else:
                # ! assume 1st return is alwasy run_id
                result = wrap_func(run_idx, func, *args)
                result = handle_wrapper_results(result)
                result_list.append(result)
                progress_bar.update()

        if num_workers > 1:
            for future in as_completed(future_list):
                if future.exception() is not None:
                    if crash_on_exception:
                        raise future.exception()
                    logging.info(future.exception())
                    continue
                result = future.result()
                result = handle_wrapper_results(result)
                result_list.append(result)
                progress_bar.update()
            executor.shutdown()
        progress_bar.close()

    # shutdown the pool, cancels scheduled tasks, returns when running tasks complete
    # if executor:
    #     executor.shutdown(wait=True, cancel_futures=True)

    result_list = sorted(result_list, key=lambda k: k[0])
    result_list = [v[1] for v in result_list]
    return result_list


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                f"{colored_word}: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode."
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


def difference_filename(listA, listB):
    """Return paths in A that dont have filename in B."""
    name_listB = [pathlib.Path(v).stem for v in listB]
    name_listB = list(set(name_listB))
    name_listA = [pathlib.Path(v).stem for v in listA]
    sel_idx_list = []
    for idx, name in enumerate(name_listA):
        try:
            name_listB.index(name)
        except ValueError:
            sel_idx_list.append(idx)
    if len(sel_idx_list) == 0:
        return []
    sublistA = np.array(listA)[np.array(sel_idx_list)]
    return sublistA.tolist()


def intersection_filename(listA, listB):
    """Return paths with file name exist in both A and B."""
    name_listA = [pathlib.Path(v).stem for v in listA]
    name_listB = [pathlib.Path(v).stem for v in listB]
    union_name_list = list(set(name_listA).intersection(set(name_listB)))
    union_name_list.sort()

    sel_idx_list = []
    for _, name in enumerate(union_name_list):
        try:
            sel_idx_list.append(name_listA.index(name))
        except ValueError:
            pass
    if len(sel_idx_list) == 0:
        return [], []
    sublistA = np.array(listA)[np.array(sel_idx_list)]

    sel_idx_list = []
    for _, name in enumerate(union_name_list):
        try:
            sel_idx_list.append(name_listB.index(name))
        except ValueError:
            pass
    sublistB = np.array(listB)[np.array(sel_idx_list)]

    return sublistA.tolist(), sublistB.tolist()


def __walk_list_dict(in_list_dict):
    """Recursive walk and jsonify in place.
    Args:
        in_list_dict (list or dict):  input list or a dictionary.
    Returns:
        list or dict
    """
    if isinstance(in_list_dict, dict):
        __walk_dict(in_list_dict)
    elif isinstance(in_list_dict, list):
        __walk_list(in_list_dict)
    elif isinstance(in_list_dict, np.ndarray):
        in_list_dict = in_list_dict.tolist()
        __walk_list(in_list_dict)
    elif isinstance(in_list_dict, np.generic):
        in_list_dict = in_list_dict.item()
    elif in_list_dict is not None and not isinstance(
        in_list_dict, (int, float, str, bool)
    ):
        raise ValueError(
            f"Value type `{type(in_list_dict)}` `{in_list_dict}` is not jsonified."
        )
    return in_list_dict


def __walk_list(lst):
    """Recursive walk and jsonify a list in place.
    Args:
        lst (list):  input list.
    """
    for i, v in enumerate(lst):
        lst[i] = __walk_list_dict(v)


def __walk_dict(dct):
    """Recursive walk and jsonify a dictionary in place.
    Args:
        dct (dict):  input dictionary.
    """
    for k, v in dct.items():
        if not isinstance(k, (int, float, str, bool)):
            raise ValueError(f"Key type `{type(k)}` `{k}` is not jsonified.")
        dct[k] = __walk_list_dict(v)


def load_json(path):
    with open(path, "r") as fptr:
        return json.load(fptr)


def save_as_json(data, save_path):
    """Save data to a json file.
    The function will deepcopy the `data` and then jsonify the content
    in place. Support data types for jsonify consist of `str`, `int`, `float`,
    `bool` and their np.ndarray respectively.
    Args:
        data (dict or list): Input data to save.
        save_path (str): Output to save the json of `input`.
    """
    shadow_data = copy.deepcopy(data)  # make a copy of source input
    if not isinstance(shadow_data, (dict, list)):
        raise ValueError(f"Type of `data` ({type(data)}) must be in (dict, list).")

    if isinstance(shadow_data, dict):
        __walk_dict(shadow_data)
    else:
        __walk_list(shadow_data)

    with open(save_path, "w") as handle:
        json.dump(shadow_data, handle, indent=4)


def load_yaml(path):
    with open(path) as fptr:
        info = yaml.full_load(fptr)
    return info


def update_nested_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_nested_dict(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        # elif isinstance(val, list):
        #     orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def get_from_nested_dict(nested_dict, nested_key_list):
    return reduce(operator.getitem, nested_key_list, nested_dict)


def flatten_dict_hierarchy(nested_key_list, raw_data):
    output_list = []
    for step_output in raw_data:
        step_output = get_from_nested_dict(step_output, nested_key_list)
        step_output = np.split(step_output, step_output.shape[0], axis=0)
        output_list.extend(step_output)
    output_list = [np.squeeze(v) for v in output_list]
    return output_list


def rgb2int(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    buffer = r.copy().astype(np.int32)
    buffer = (buffer << 8) + g
    buffer = (buffer << 8) + b
    return buffer


def save_as_yaml(data, save_path):
    with open(save_path, "w") as fptr:
        yaml.dump(data, fptr, default_flow_style=False)


def set_logger(path):
    logging.basicConfig(level=logging.INFO)
    # * reset logger handler
    log_formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    new_hdlr_list = [logging.FileHandler(path), logging.StreamHandler()]
    for hdlr in new_hdlr_list:
        hdlr.setFormatter(log_formatter)
        log.addHandler(hdlr)


def convert_zip_to_npy(zip_file):
    npy_stem = pathlib.Path(zip_file).stem
    npy_file = str(pathlib.Path(zip_file).parent) + f"/{npy_stem}.npy"
    temp_path = f"{pathlib.Path(zip_file).parent}/tmp/"
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_path)

    files = os.listdir(temp_path)
    files = [f"{temp_path}/{f}" for f in files]
    files.sort()
    images = np.empty((len(files), 1024, 1024, 3), dtype=np.uint8)
    for idx, image in enumerate(files):
        if image.endswith(".png"):
            images[idx] = imread(image)
    np.save(npy_file, images)
    rmdir(temp_path)
    os.remove(zip_file)
    return


def create_image_transform(
    input_resolution, output_size, output_resolution, is_tsf=False
):
    """Creates transform to resize, rescale, convert to tensor and put in range [-1, 1] or [0, 1]"""
    # assert input_resolution in ["80x", "40x", "20x"]
    assert input_resolution in ["40x", "20x"]
    assert output_size in [1024, 512, 256, 128, 64]

    transform = [T.ToTensor()]

    if output_resolution == "40x":
        if output_size != 1024:
            transform.append(T.CenterCrop(output_size))
        else:
            pass
    elif output_resolution == "20x":
        if output_size == 512:
            transform.append(T.Resize(512, antialias=True))
        else:
            transform.append(T.CenterCrop(int(output_size * 2)))
            transform.append(T.Resize(output_size, antialias=True))
    elif output_resolution == "80x":
        transform.append(T.CenterCrop(int(output_size / 2)))
        transform.append(T.Resize(output_size, antialias=True))

    if is_tsf:
        transform.append(T.Normalize([0.5], [0.5]))

    transform = T.Compose(transform)
    return transform


def inverse_transform(tensor, return_numpy=False, is_tsf=False):
    """
    Convert transformed images' tensors from [-1., 1.] to [0., 255.]
    and convert source and target images' tensors from [0., 1.] to [0., 255.]
    """
    if is_tsf:
        new_tensor = ((tensor.clamp(-1, 1) + 1.0) / 2.0) * 255.0
    else:
        new_tensor = tensor.clamp(0, 1) * 255.0
    if new_tensor.dim() == 4:
        new_tensor = new_tensor.permute(0, 2, 3, 1)
    elif new_tensor.dim() == 3:
        new_tensor = new_tensor.permute(1, 2, 0)
    if not return_numpy:
        return new_tensor.type(torch.uint8)
    else:
        return new_tensor.cpu().numpy().astype(np.uint8)


def get_type_from_config(target_key):
    module_name, class_name = target_key.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    config_class = getattr(module, class_name)
    return config_class


def get_kwargs_from_config(config_kwargs):
    config_kwargs = dict(config_kwargs)
    config_kwargs.pop("_target_")
    return config_kwargs


def save_image(img, output_dir, stem, ext):
    """Helper function for multiprocessing saving images"""
    file_name = f"{output_dir}/{stem}.{ext}"
    imwrite(file_name, img)
    return None


def create_target_inf(target_path, size=512):
    target = imread(target_path)
    if size != target.shape[0]:
        target = cv2.resize(target, (size, size), interpolation=cv2.INTER_NEAREST)
    transform = T.Compose([T.ToTensor()])
    target_t = transform(target).type(torch.float32)
    return target_t


def rgba_to_int(r: int, g: int, b: int, a: int = 1) -> int:
    """Generate int color value for OME TIFF.
    Use int.from_bytes to convert a color tuple.
    >>> print(rgba_to_int(0, 0, 0, 0))
    0
    >>> print(rgba_to_int(0, 1, 135, 4))
    100100
    """
    return int.from_bytes([r, g, b, a], byteorder="big", signed=True)


def npy2pyramid(
    save_path: str,
    image: np.ndarray,
    mode: str = "rgb",
    channels=None,
    pyramid: dict = None,
    resolution: float = 0.25,
) -> None:
    """
    Args:
        channels (list): A list of dict where each contains the metadata
            for each channel. The dict key must be an attribute following
            OME-TIFF convention. Each dict metadata within the list corresponds
            to the channel at the same location in the input image. For example

            >>> channel_0 = {"color": (255, 0, 255), "name": "CD4"}
            >>> channel_1 = {"color": (  0, 0, 255), "name": "CD8"}
            >>> channels = [channel_0, channel_1]

        pyramid (dict): A dict of keyword parameters for saving the vips image.
            If it is set to `None`, the following default paramters are used.
            >>> default_paramters = dict(
            >>>     compression="lzw",
            >>>     compression='jpeg',
            >>>     Q=80,
            >>>     tile=True,
            >>>     tile_width=256,
            >>>     tile_height=256,
            >>>     pyramid=True,
            >>> )
        resolution (float): um/pixel that the canvas to be saved is currently at.
    """
    import pyvips

    # will crash if the input image
    # is smaller than the tile size
    default_pyramid = dict(
        compression="jpeg",
        Q=80,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
    )

    pyramid = default_pyramid if pyramid is None else pyramid
    np_dtype_to_vip_dtype = {
        "uint8": "uchar",
        "int8": "char",
        "uint16": "ushort",
        "int16": "short",
        "uint32": "uint",
        "int32": "int",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "dpcomplex",
    }

    vi_dtype = np_dtype_to_vip_dtype[str(image.dtype)]
    image_shape = image.shape
    assert mode in ["rgb", "multiplex"]
    assert len(image_shape) in [2, 3]

    if len(image_shape) == 2:
        h, w = image_shape
        c = 1
    else:
        h, w, c = image_shape

    # `bands` is pyvips's terminology for channels
    image_ = image.reshape(h * w * c)

    vi = pyvips.Image.new_from_memory(image_.data, w, h, c, vi_dtype)

    if resolution is not None:
        vi.set_type(pyvips.GValue.gint_type, "page-height", h)
        resolution = (resolution, resolution)
        pixel_per_micron = 1 / np.array(resolution)
        pixel_per_centimeter = pixel_per_micron * 1.0e3
        vi.set_type(pyvips.GValue.guint64_type, "xres", 1000)
        vi.set_type(pyvips.GValue.guint64_type, "yres", 1000)
        vi.set_type(pyvips.GValue.gstr_type, "resolution-unit", "cm")
        vi = vi.copy(
            xres=int(pixel_per_centimeter[0]), yres=int(pixel_per_centimeter[1])
        )

    vi.tiffsave(save_path, **pyramid)
    return
