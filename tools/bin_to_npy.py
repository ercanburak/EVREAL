import os
import json
import glob
import argparse

import numpy as np


def bin_to_npy(bag_path, output_pth, width=34, height=34):
    # Adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py
    events_ts_path = os.path.join(output_pth, 'events_ts.npy')
    events_xy_path = os.path.join(output_pth, 'events_xy.npy')
    events_p_path = os.path.join(output_pth, 'events_p.npy')

    f = open(bag_path, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process timestamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    xs = all_x[td_indices]
    ys = all_y[td_indices]
    ps = all_p[td_indices]
    ts = all_ts[td_indices]

    events_ts = np.array(ts).astype(np.float64) / 1e6  # Convert to seconds
    events_xy = np.array([xs, ys]).transpose().astype(np.int16)
    events_p = np.array(ps).astype(bool)

    np.save(events_ts_path, events_ts)
    np.save(events_xy_path, events_xy)
    np.save(events_p_path, events_p)


if __name__ == "__main__":
    """ Tool for converting rosbag events and images to numpy format. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Root directory of dataset with classes and class instances as "
                                     "binary files (e.g. NCaltech101, NMNIST, etc.)")
    parser.add_argument("--width", type=int, default=34, help="Width of the event sensor")
    parser.add_argument("--height", type=int, default=34, help="Height of the event sensor")
    parser.add_argument("--remove", help="Remove rosbags after conversion", action="store_true")
    args = parser.parse_args()
    bin_paths = sorted(glob.glob(os.path.join(args.path, "*", "*.bin")))
    for path in bin_paths:
        print(f"Processing {path}")
        output_pth = os.path.splitext(path)[0]
        os.makedirs(output_pth, exist_ok=True)
        bin_to_npy(path, output_pth, width=args.width, height=args.height)
        if args.remove:
            os.remove(path)
    print("Done.")
