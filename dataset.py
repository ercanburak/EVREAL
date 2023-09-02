import os
import warnings
from bisect import bisect_left

import numpy as np
import torch
from torch.utils.data import Dataset

# local modules
from utils.event_utils import events_to_voxel_torch
from utils.util import read_json


class MemMapDataset(Dataset):

    def __init__(self, data_path, sensor_resolution=None, num_bins=5,
                 voxel_method=None, max_length=None, keep_ratio=1):
        self.num_bins = num_bins
        self.data_path = data_path
        self.keep_ratio = keep_ratio
        self.sensor_resolution = sensor_resolution
        self.has_images = True
        self.channels = self.num_bins
        self.load_data(data_path)
        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.voxel_method = voxel_method
        self.set_voxel_method()

        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    def __getitem__(self, index):

        if self.voxel_method['method'] == 'between_frames':
            assert 0 <= index < self.__len__(), f"index {index} out of bounds (0 <= x < {self.__len__()})"
            if index > 0:
                prev_index = self.frames_to_use[index - 1]
            else:
                prev_index = 0
            index = self.frames_to_use[index]
            _, idx0 = self.get_event_indices(prev_index)
            _, idx1 = self.get_event_indices(index)
        else:
            assert 0 <= index < self.__len__(), f"index {index} out of bounds (0 <= x < {self.__len__()})"
            idx0, idx1 = self.get_event_indices(index)

        xs, ys, ts, ps = self.get_events(idx0, idx1)

        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0

        event_count = len(xs)
        if event_count < 3:
            voxel = self.get_empty_voxel_grid()
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts - ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))
            voxel = self.get_voxel_grid(xs, ys, ts, ps)

        if self.voxel_method['method'] == 't_seconds':
            dt = self.voxel_method['t']
        else:
            dt = ts_k - ts_0
        if dt == 0:
            dt = np.array(0.0)

        if self.has_images and self.voxel_method['method'] != 'between_frames':
            index = self.get_closest_frame_index(ts_k)

        if self.has_images:
            frame = self.get_frame(index)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
            frame_timestamp = torch.tensor(self.frame_ts[index], dtype=torch.float64)
        else:
            frame = torch.zeros((1, self.sensor_resolution[0], self.sensor_resolution[1]),
                                dtype=torch.float32, device=voxel.device)
            frame_timestamp = torch.tensor(0.0, dtype=torch.float64)

        voxel_timestamp = torch.tensor(ts_k, dtype=torch.float64)

        item = {'frame': frame,
                'events': voxel,
                'frame_timestamp': frame_timestamp,
                'voxel_timestamp': voxel_timestamp,
                'dt': torch.tensor(dt, dtype=torch.float64),
                'event_count': event_count}
        return item

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(len(self)):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        """
        k_indices = []
        start_idx = 0
        for i in range(len(self)):
            idx0 = (self.voxel_method['k'] - self.voxel_method['sliding_window_w']) * i
            idx1 = idx0 + self.voxel_method['k']
            k_indices.append([idx0, idx1])
        return k_indices

    def choose_frames_to_use(self):
        self.frames_to_use = list(range(0, self.num_frames))
        if self.keep_ratio != 1:
            assert self.voxel_method['method'] == 'between_frames', \
                "keep_ratio can only specified for between_frames voxel method"
            assert self.keep_ratio < 1, "keep_ratio cannot be greater than 1"
            num_frames_to_use = int(self.num_frames * self.keep_ratio)
            self.frames_to_use = sorted(np.random.choice(self.frames_to_use, size=num_frames_to_use, replace=False))
            self.length = num_frames_to_use - 1

    def get_min_max_t(self):
        if self.has_images:
            min_t = min(self.frame_ts[0], self.t0)
            max_t = max(self.frame_ts[-1], self.tk)
        else:
            min_t = self.t0
            max_t = self.tk
        return min_t, max_t

    def get_closest_frame_index(self, ts):
        pos = bisect_left(self.frame_ts, ts)
        if pos == 0:
            # return self.get_frame(0)
            return 0
        if pos == len(self.frame_ts):
            # return self.get_frame(-1)
            return pos - 1
        before = self.frame_ts[pos - 1]
        after = self.frame_ts[pos]
        if after - ts < ts - before:
            # return self.get_frame(pos)
            return pos

        # return self.get_frame(pos - 1)
        return pos - 1

    def set_voxel_method(self):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events / (self.voxel_method['k'] - self.voxel_method['sliding_window_w'])), 0)
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            duration = self.tk - self.t0
            self.length = max(int(duration / (self.voxel_method['t'] - self.voxel_method['sliding_window_t'])), 0)
            self.event_indices = self.compute_timeblock_indices()
        elif self.voxel_method['method'] == 'between_frames':
            assert self.has_images, "Cannot use between_frames voxel method without images"
            self.length = self.num_frames - 1
            self.event_indices = self.compute_frame_indices()
            self.choose_frames_to_use()
        else:
            raise ValueError("Invalid voxel forming method chosen ({})".format(self.voxel_method))

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise ValueError("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_empty_voxel_grid(self):
        """Return an empty voxel grid filled with zeros"""
        size = (self.num_bins, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        create voxel grid merging positive and negative events (resulting in NUM_BINS x H x W tensor).
        """
        # generate voxel grid which has size self.num_bins x H x W
        voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        return voxel_grid

    def get_frame(self, index):
        frame = self.filehandle['images'][index][:, :, 0]
        return frame

    def get_events(self, idx0, idx1):
        xy = self.filehandle["xy"][idx0:idx1]
        xs = xy[:, 0].astype(np.float32)
        ys = xy[:, 1].astype(np.float32)
        ts = self.filehandle["t"][idx0:idx1]
        ps = self.filehandle["p"][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path):
        assert os.path.isdir(data_path), f'{data_path} is not a valid data_path'
        data = {}
        events_ts_path = os.path.join(data_path, 'events_ts.npy')
        events_xy_path = os.path.join(data_path, 'events_xy.npy')
        events_p_path = os.path.join(data_path, 'events_p.npy')
        images_path = os.path.join(data_path, 'images.npy')
        images_ts_path = os.path.join(data_path, 'images_ts.npy')
        image_event_indices_path = os.path.join(data_path, 'image_event_indices.npy')

        if os.path.exists(images_ts_path) and os.path.exists(images_path) and os.path.exists(image_event_indices_path):
            data["frame_stamps"] = np.load(images_ts_path)
            data["images"] = np.load(images_path, mmap_mode='r')
            data["image_event_indices"] = np.load(image_event_indices_path)
            self.has_images = True
        else:
            self.has_images = False

        data["t"] = np.load(events_ts_path, mmap_mode='r').squeeze()
        data["xy"] = np.load(events_xy_path, mmap_mode='r').squeeze()
        data["p"] = np.load(events_p_path, mmap_mode='r').squeeze()

        data['path'] = data_path
        assert (len(data['p']) == len(data['xy']) and len(data['p']) == len(data['t'])), \
            "Number of events, timestamps and coordinates do not match"

        self.t0, self.tk = data['t'][0], data['t'][-1]
        self.num_events = len(data['p'])

        self.frame_ts = []
        if self.has_images:
            self.num_frames = len(data['images'])
            for ts in data["frame_stamps"]:
                self.frame_ts.append(ts.item())
            data["index"] = self.frame_ts
        else:
            self.num_frames = 0

        assert (len(self.frame_ts) == self.num_frames), "Number of frames and timestamps do not match"
        self.filehandle = data
        if self.sensor_resolution is None:
            metadata_path = os.path.join(data_path, "metadata.json")
            if os.path.exists(metadata_path):
                metadata = read_json(metadata_path)
                self.sensor_resolution = metadata["sensor_resolution"]
            else:
                # get sensor resolution from data
                if self.has_images and self.num_frames > 0:
                    self.sensor_resolution = self.filehandle["images"][0].shape[:2]
                else:
                    self.sensor_resolution = [np.max(self.filehandle["xy"][:, 1]) + 1,
                                              np.max(self.filehandle["xy"][:, 0]) + 1]

    def find_ts_index(self, timestamp):
        index = np.searchsorted(self.filehandle["t"], timestamp)
        return index

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for event_idx in self.filehandle["image_event_indices"]:
            end_idx = event_idx[0]
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices
