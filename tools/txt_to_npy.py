import os
import json
import glob
import argparse

import numpy as np


def txt_to_npy(input_path, output_path, flip=False):
    with open(input_path) as f:
        sensor_size_str = f.readline().strip('\n')
    width, height = [int(x) for x in sensor_size_str.split()]
    sensor_size = height, width
    data = np.loadtxt(input_path,  dtype=np.float64, skiprows=1)
    xy_path = os.path.join(output_path, 'events_xy.npy')
    t_path = os.path.join(output_path, 'events_ts.npy')
    p_path = os.path.join(output_path, 'events_p.npy')
    event_ts = data[:, 0].copy()
    events_xy = data[:, 1:3].copy()
    if flip:
        events_xy[:, 0] = width - 1 - events_xy[:, 0]
        events_xy[:, 1] = height - 1 - events_xy[:, 1]
    events_polar = data[:, 3].copy()
    event_ts = event_ts.astype('float64')
    events_xy = events_xy.astype('int16')
    events_polar = events_polar.astype('bool')
    np.save(xy_path, events_xy, allow_pickle=False, fix_imports=False)
    np.save(t_path, event_ts, allow_pickle=False, fix_imports=False)
    np.save(p_path, events_polar, allow_pickle=False, fix_imports=False)

    # write sensor size to metadata
    if sensor_size is not None:
        metadata_path = os.path.join(output_pth, 'metadata.json')
        metadata = {"sensor_resolution": sensor_size}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)


if __name__ == "__main__":
    """ Tool for converting event txt files to numpy format. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory of event txt files")
    parser.add_argument("--flip", help="Flip x and y axis", action="store_true")
    parser.add_argument("--remove", help="Remove txt files after conversion", action="store_true")
    args = parser.parse_args()
    txt_paths = sorted(glob.glob(os.path.join(args.path, "*.txt")))
    for path in txt_paths:
        print("Processing {}".format(path))
        output_pth = os.path.splitext(path)[0]
        os.makedirs(output_pth, exist_ok=True)
        txt_to_npy(path, output_pth, flip=args.flip)
        if args.remove:
            os.remove(path)
