import argparse
import os
import cv2
import numpy as np
import glob
import shutil

FRAME_WIDTH = 970
FRAME_HEIGHT = 625


def convert_and_fix_event_pixels(data, upper_limit, fix_overflows=True):
    data = data.astype(np.int32)
    overflow_indices = np.where(data > upper_limit*32)
    num_overflows = overflow_indices[0].shape[0]
    if fix_overflows and num_overflows > 0:
        data[overflow_indices] = data[overflow_indices] - 65536
    data = data / 32.0
    data = np.rint(data)
    data = data.astype(np.int16)
    data = np.clip(data, 0, upper_limit)
    return data


def bs_ergb_to_memmap(input_path, output_path):
    events_ts_path = os.path.join(output_path, 'events_ts.npy')
    events_xy_path = os.path.join(output_path, 'events_xy.npy')
    events_p_path = os.path.join(output_path, 'events_p.npy')
    images_path = os.path.join(output_path, 'images.npy')
    images_ts_path = os.path.join(output_path, 'images_ts.npy')
    image_event_indices_path = os.path.join(output_path, 'image_event_indices.npy')
    images_folder = os.path.join(input_path, 'images')
    events_folder = os.path.join(input_path, 'events')
    image_files_glob_pattern = os.path.join(images_folder, "*.png")
    event_files_glob_pattern = os.path.join(events_folder, "*.npz")
    image_timestamps_file_path = os.path.join(images_folder, "timestamp.txt")
    image_file_paths = sorted(glob.glob(image_files_glob_pattern))
    event_file_paths = sorted(glob.glob(event_files_glob_pattern))
    len_images = len(image_file_paths)
    assert len_images == len(event_file_paths) + 1
    os.makedirs(output_path)
    start_timestamp_s = 0.0

    # Save images_ts
    image_ts_list = []
    with open(image_timestamps_file_path) as f:
        file_lines = f.readlines()
        start_timestamp_s = float(file_lines[0]) / 1000000.0
        for line in file_lines:
            frame_ts_s = float(line) / 1000000.0
            image_ts_normalized = np.array(frame_ts_s - start_timestamp_s)
            image_ts_normalized = np.expand_dims(image_ts_normalized, axis=-1)
            image_ts_normalized = np.expand_dims(image_ts_normalized, axis=0)
            image_ts_list.append(image_ts_normalized)
    images_ts_data = np.concatenate(image_ts_list)
    np.save(images_ts_path, images_ts_data, allow_pickle=False, fix_imports=False)
    del images_ts_data
    del image_ts_list

    # Save image_event_indices
    first_idx = np.zeros((1, 1), dtype=np.int64)
    image_event_indices = [first_idx]
    total_event_num = 0
    for event_file_path in event_file_paths:
        event_file_data = np.load(event_file_path)
        event_num = event_file_data['x'].shape[0]
        total_event_num = total_event_num + event_num
        event_idx = np.array(total_event_num)
        event_idx = np.expand_dims(event_idx, axis=-1)
        event_idx = np.expand_dims(event_idx, axis=0)
        image_event_indices.append(event_idx)
    image_event_indices = np.concatenate(image_event_indices)
    np.save(image_event_indices_path, image_event_indices, allow_pickle=False, fix_imports=False)

    # Save events
    x_data = np.zeros(shape=total_event_num, dtype=np.uint16)
    y_data = np.zeros(shape=total_event_num, dtype=np.uint16)
    t_data = np.zeros(shape=total_event_num, dtype=np.uint32)
    p_data = np.zeros(shape=total_event_num, dtype=np.uint8)
    for frame_idx, event_file_path in enumerate(event_file_paths):
        start_event_idx = image_event_indices[frame_idx].item()
        end_event_idx = image_event_indices[frame_idx+1].item()
        event_file_data = np.load(event_file_path)
        x_data[start_event_idx:end_event_idx] = convert_and_fix_event_pixels(event_file_data['x'], FRAME_WIDTH - 1)
        y_data[start_event_idx:end_event_idx] = convert_and_fix_event_pixels(event_file_data['y'], FRAME_HEIGHT - 1)
        t_data[start_event_idx:end_event_idx] = event_file_data['timestamp']
        p_data[start_event_idx:end_event_idx] = event_file_data['polarity']
    xy_data = np.stack([x_data, y_data], axis=-1)
    t_data = t_data.astype(np.float64)
    t_data = t_data / 1000000.0  # convert us to s
    t_data = t_data - start_timestamp_s  # zeroize timestamps
    np.save(events_ts_path, t_data, allow_pickle=False, fix_imports=False)
    np.save(events_xy_path, xy_data, allow_pickle=False, fix_imports=False)
    np.save(events_p_path, p_data, allow_pickle=False, fix_imports=False)
    del x_data
    del y_data
    del t_data
    del p_data

    # Save images
    images_list = []
    for image_file_path in image_file_paths:
        img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        images_list.append(img)
    images_data = np.concatenate(images_list)
    np.save(images_path, images_data, allow_pickle=False, fix_imports=False)


if __name__ == "__main__":
    """
    Tool to convert BS_ERGB dataset files to the memmap format
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to sequence folder with subfolders for events and images")
    parser.add_argument("--output_dir", help="Path to extract", required=True)
    parser.add_argument("--overwrite", help="Overwrite existing folders", action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print("Output directory must be a directory")
        exit()

    seq_name = os.path.basename(os.path.normpath(args.path))
    output_path = os.path.join(args.output_dir, seq_name)

    if os.path.exists(output_path):
        if not args.overwrite:
            print("Output directory {} already exists, exiting".format(output_path))
            exit()
        else:
            shutil.rmtree(output_path)

    bs_ergb_to_memmap(args.path, output_path)
