import os
import json
import glob
import argparse

import numpy as np
import embag as rosbag
from tqdm import tqdm
from cv_bridge import CvBridge


def timestamp_float(ts):
    return ts.secs + ts.nsecs / float(1e9)


def bag_to_npy(bag_path, output_pth, event_topic, image_topic):
    metadata_path = os.path.join(output_pth, 'metadata.json')
    events_ts_path = os.path.join(output_pth, 'events_ts.npy')
    events_xy_path = os.path.join(output_pth, 'events_xy.npy')
    events_p_path = os.path.join(output_pth, 'events_p.npy')
    images_path = os.path.join(output_pth, 'images.npy')
    images_ts_path = os.path.join(output_pth, 'images_ts.npy')
    image_event_indices_path = os.path.join(output_pth, 'image_event_indices.npy')
    sensor_size = None
    topics = [image_topic, event_topic]
    xs, ys, ts, ps = [], [], [], []
    image_list, image_ts_list, image_event_indices_list = [], [], []
    bag = rosbag.Bag(bag_path)
    for topic, msg, t in tqdm(bag.read_messages(topics)):
        if topic == event_topic:
            for e in msg.events:
                timestamp = timestamp_float(e.ts)
                xs.append(e.x)
                ys.append(e.y)
                ts.append(timestamp)
                ps.append(1 if e.polarity else 0)
        elif topic == image_topic:
            timestamp = timestamp_float(msg.header.stamp)
            image_ts_list.append(timestamp)
            image = CvBridge().imgmsg_to_cv2(msg, "mono8")
            image_list.append(image)
            if sensor_size is None:
                sensor_size = image.shape[:2]
    bag.close()

    events_ts = np.array(ts)
    events_xy = np.array([xs, ys]).transpose()
    events_p = np.array(ps)
    # assert np.all(events_ts[:-1] <= events_ts[1:])

    images = np.stack(image_list)
    images = np.expand_dims(images, axis=-1)
    images_ts = np.expand_dims(np.stack(image_ts_list), axis=1)
    # assert np.all(images_ts[:-1] <= images_ts[1:])

    # zero timestamps
    img_min_ts = np.min(images_ts)
    events_min_ts = np.min(events_ts)
    min_ts = min(img_min_ts, events_min_ts)
    events_ts -= min_ts
    images_ts -= min_ts
    # assert np.all(events_ts >= 0)
    # assert np.all(images_ts >= 0)

    # calculate image_event_indices
    image_event_indices = np.searchsorted(events_ts, images_ts, 'right') - 1
    image_event_indices = np.clip(image_event_indices, 0, len(events_ts) - 1)

    np.save(events_ts_path, events_ts, allow_pickle=False, fix_imports=False)
    np.save(events_xy_path, events_xy, allow_pickle=False, fix_imports=False)
    np.save(events_p_path, events_p, allow_pickle=False, fix_imports=False)

    np.save(images_path, images, allow_pickle=False, fix_imports=False)
    np.save(images_ts_path, images_ts, allow_pickle=False, fix_imports=False)
    np.save(image_event_indices_path, image_event_indices, allow_pickle=False, fix_imports=False)

    # write sensor size to metadata
    metadata = {"sensor_resolution": sensor_size}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    """ Tool for converting rosbag events and images to numpy format. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory of ROS bags")
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic")
    parser.add_argument("--image_topic", default="/dvs/image_raw", help="Image topic")
    parser.add_argument("--remove", help="Remove rosbags after conversion", action="store_true")
    args = parser.parse_args()
    rosbag_paths = sorted(glob.glob(os.path.join(args.path, "*.bag")))
    for path in rosbag_paths:
        print("Processing {}".format(path))
        output_pth = os.path.splitext(path)[0]
        os.makedirs(output_pth, exist_ok=True)
        bag_to_npy(path, output_pth, args.event_topic, args.image_topic)
        if args.remove:
            os.remove(path)
