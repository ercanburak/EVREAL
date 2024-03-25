import os
from pathlib import Path
import sys
""
import numpy as np
import cv2


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def extract_gt_images(input_dir, output_dir):
    images_array_path = os.path.join(input_dir, 'images.npy')
    images_array = np.load(images_array_path)
    ensure_dir(output_dir)
    for idx, image in enumerate(images_array):
        image_path = os.path.join(output_dir, 'frame_{:010d}.png'.format(idx))
        image = np.squeeze(image)
        cv2.imwrite(image_path, image)


extract_gt_images(sys.argv[1], sys.argv[2])
