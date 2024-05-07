import os
import shutil
import glob
from tqdm import tqdm

from utils.eval_utils import ensure_dir


glob_pattern = os.path.join("outputs", "t60ms_s", "NCaltech101", "*", "*")
folder_paths = sorted(glob.glob(glob_pattern))
for folder_path in tqdm(folder_paths):
    folder_path = os.path.normpath(folder_path)
    image_file_path = os.path.join(folder_path, "frame_0000000002.png")
    if not os.path.isfile(image_file_path):
        print("Could not find image at " + image_file_path)
        continue
    folder_path_parts = folder_path.split(os.sep)
    model_name = folder_path_parts[-1]
    class_and_instance_name = folder_path_parts[-2]
    class_name, instance_name = class_and_instance_name.split("_image_")
    new_folder_path = os.path.join("outputs",  "NCaltech101", model_name, class_name)
    ensure_dir(new_folder_path)
    new_image_path = os.path.join(new_folder_path, instance_name + ".png")
    shutil.copy2(image_file_path, new_image_path)
