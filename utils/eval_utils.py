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

#---------
def save_flow(folder, flow, idx):
    png_name = 'frame_{:010d}.png'.format(idx)
    png_path = join(folder, png_name)
    cv2.imwrite(png_path, flow)


def make_event_preview(events, mode='grayscale', num_bins_to_show=-1):
    # events: [1 x C x H x W] event numpy or [C x H x W]
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    if events.ndim == 3:
        events = np.expand_dims(events,axis=0)
    if num_bins_to_show < 0:
        sum_events = np.sum(events[0, :, :, :], axis=0)
        # sum_events = np.sum(events[0, :, :, :], axis=0)
    else:
        sum_events = np.sum(events[0, -num_bins_to_show:, :, :], axis=0)

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        
        b[sum_events > 0] = 255 #np.clip((255.0 * (sum_events[sum_events>0] - m) / (M - m)), 0, 255).astype(np.uint8) #255
        r[sum_events < 0] = 255 #np.clip((255.0 * (sum_events[sum_events<0] - m) / (M - m)), 0, 255).astype(np.uint8) #255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        
        m, M = -5.0, 5.0 #-5.0, 5.0
        # M = (sum_events.max() - sum_events.min())/2
        # m = -M
        
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)), 0, 255).astype(np.uint8)
        # event_preview = np.clip((255.0 * (sum_events - sum_events.min()) / (sum_events.max() - sum_events.min())).astype(np.uint8), 0, 255)

    return event_preview

def save_events(folder, event_images, idx):
    #event_images: dict: voxel_grid / voxel_grid_warped
    if 'voxel_grid' in event_images.keys():
        evs= event_images['voxel_grid']
        # print('evs', evs.max(), evs.min())
        event_img = make_event_preview(evs.cpu().data.numpy(), mode='grayscale', num_bins_to_show=-1) #'red-blue'
        png_name = 'frame_{:010d}.png'.format(idx)
        png_path = join(folder, png_name)
        cv2.imwrite(png_path, event_img)
    if 'voxel_grid_warped' in event_images.keys():
        warped_evs= event_images['voxel_grid_warped']
        # print('warped_evs', warped_evs.max(), warped_evs.min())
        event_img = make_event_preview(warped_evs.cpu().data.numpy(), mode='grayscale', num_bins_to_show=-1) #'red-blue'
        png_name = 'frame_{:010d}_warped.png'.format(idx)
        png_path = join(folder, png_name)
        cv2.imwrite(png_path, event_img)