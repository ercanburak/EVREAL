from os.path import join
from pathlib import Path

import cv2
import numpy as np
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def robust_min(img, p=5):
    return np.percentile(img.ravel(), p)


def robust_max(img, p=95):
    return np.percentile(img.ravel(), p)


def normalize(img, q_min=10, q_max=90):
    """
    robust min/max normalization if specified with norm arguments
    q_min and q_max are the min and max quantiles for the normalization
    :param img: Input image to be normalized
    :param q_min: min quantile for the robust normalization
    :param q_max: max quantile for the robust normalization
    :return: Normalized image
    """
    norm_min = robust_min(img, q_min)
    norm_max = robust_max(img, q_max)
    normalized = (img - norm_min) / (norm_max - norm_min)
    return normalized


def torch2cv2(image):
    image = torch.squeeze(image)  # H x W
    image = image.cpu().numpy()
    return image


def cv2torch(image, num_ch=1):
    img_tensor = torch.tensor(image)
    if len(img_tensor.shape) == 2:
        img_tensor = torch.unsqueeze(img_tensor, 0)
        if num_ch > 1:
            img_tensor = img_tensor.repeat(num_ch, 1, 1)
    if len(img_tensor.shape) == 3:
        img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def append_timestamp(path, description, timestamp):
    with open(path, 'a', encoding="utf-8") as f:
        f.write('{} {:.15f}\n'.format(description, timestamp))


def append_result(path, description, result, is_int=False):
    format_str = '{} {}\n' if is_int else '{} {:.5f}\n'
    with open(path, 'a', encoding="utf-8") as f:
        if isinstance(result, list):
            for idx, elem in zip(description, result):
                f.write(format_str.format(idx, elem))
        else:
            f.write(format_str.format(description, result))


def setup_output_folder(output_folder):
    """
    Ensure existence of output_folder and print message
    """
    ensure_dir(output_folder)
    print('Saving results to: {}'.format(output_folder))


def save_inferred_image(folder, image, idx):
    png_name = 'frame_{:010d}.png'.format(idx)
    png_path = join(folder, png_name)
    image_for_png = np.round(image * 255).astype(np.uint8)
    cv2.imwrite(png_path, image_for_png)
