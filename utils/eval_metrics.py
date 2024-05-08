import math
import shutil
import traceback
from os.path import join
import os
import warnings
import torch.nn.functional as F

import cv2
import numpy as np
import torch
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.create_vid import create_vid_from_recon_folder
from utils.eval_utils import cv2torch, torch2cv2
from utils.eval_utils import append_timestamp, append_result, ensure_dir, save_inferred_image, save_flow, save_events

# ------ For flow --------
def merge_optical_flow(flow):
    '''Merge flow in HSV to show'''
    
    flow_x, flow_y = flow[0,...], flow[1,...]
    h, w = flow_x.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = 255
    hsv[..., 1] = 255

    # Convert flow to polar coordinates (magnitude, angle)
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    # Map angle to hue
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Normalize magnitude to 0-255
    # hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = (255 * magnitude / magnitude.max()).astype(np.uint8)
    # Convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # rgb = cv2.applyColorMap(rgb, cv2.COLORMAP_PARULA) #COLORMAP_PARULA cv2.COLORMAP_SPRING cv2.COLORMAP_COOL
    return rgb


# -------- For FWL, real data w/o real flow ----
def voxel_warping_flow_loss(voxel, displacement, output_images=False, reverse_time=False):
    """ 
    Adapted from
    Stoffregen, Timo, et al. "Reducing the sim-to-real gap for event cameras." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVII 16. Springer International Publishing, 2020.
    https://github.com/TimoStoff/event_cnn_minimal
    
    Adapted from:
        Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        This function takes an optic flow tensor and uses this to warp the channels of an
        event voxel grid to an image. The variance of this image is the resulting loss.
        :param voxel: [N x C x H x W] input voxel
        :param displacement: [N x 2 x H x W] displacement map from previous flow tensor to current flow tensor
    """
    if reverse_time:
        displacement = -displacement
    # if displacement.all():
    #     voxel_grid_warped_save = voxel.detach()
    #     voxel_grid_warped = voxel.sum(1)
    # else:
    v_shape = voxel.size()
    t_width, t_height, t_channels = v_shape[3], v_shape[2], v_shape[1]
    yy, xx = torch.meshgrid(torch.arange(t_height), torch.arange(t_width))  # xx, yy -> WxH
    xx, yy = xx.to(voxel.device).float(), yy.to(voxel.device).float()

    displacement_x = displacement[:, 0, :, :]  # N x H x W
    displacement_y = displacement[:, 1, :, :]  # N x H x W
    displacement_increment = 1.0/(t_channels-1.0)
    voxel_grid_warped = torch.zeros((v_shape[0], 1, t_height, t_width), dtype=voxel.dtype, device=voxel.device) 
    
    voxel_grid_warped_save = torch.zeros((v_shape[0], t_channels, t_height, t_width), dtype=voxel.dtype, device=voxel.device) 
    for i in range(t_channels):
        warp_magnitude_ratio = (1.0-i*displacement_increment) if reverse_time else i*displacement_increment
        #Add displacement to the x coords
        warping_grid_x = xx + displacement_x*warp_magnitude_ratio # N x H x W
        #Add displacement to the y coords
        warping_grid_y = yy + displacement_y*warp_magnitude_ratio # N x H x W
        warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2
        #Normalize the warping grid to between -1 and 1 (necessary for grid_sample API)
        warping_grid[:,:,:,1] = (2.0*warping_grid[:,:,:,1])/(t_height)-1.0
        warping_grid[:,:,:,0] = (2.0*warping_grid[:,:,:,0])/(t_width)-1.0
        voxel_channel_warped = F.grid_sample(voxel, warping_grid,  align_corners=True) #, padding_mode='reflection'
        voxel_grid_warped+=voxel_channel_warped[:, i:i+1, :, :].detach()
        voxel_grid_warped_save[:, i:i+1, :,:] = voxel_channel_warped[:, i:i+1, :, :].detach()

    variance = voxel_grid_warped.var()
    tc_loss = variance
    # tc_loss = -variance
    # if not reverse_time:
    #     reverse_tc_loss = voxel_warping_flow_loss(voxel, displacement, output_images=False, reverse_time=True)
    #     tc_loss += reverse_tc_loss
    if output_images:
        additional_output = {'voxel_grid': voxel,
                             'voxel_grid_warped': voxel_grid_warped_save} #voxel_grid_warped
        return tc_loss, additional_output
    else:
        return tc_loss


class BaseMetric:
    """Base class for quantitative evaluation metrics"""

    def __init__(self, name, no_ref=False):
        self.scores = []
        self.name = name
        self.no_ref = no_ref
        self.updated = 0
        self.image_queue = []
        self.ref_queue = []
        self.batch_size = 4

    def reset(self):
        self.scores = []
        self.image_queue = []
        self.ref_queue = []
        self.updated = 0

    def finish_queue(self):
        self.updated = 0

    def get_num_updated(self):
        return self.updated

    def calculate(self, img, ref):
        raise NotImplementedError

    def update(self, img, ref=None):
        self.updated = 0
        score = self.calculate(img, ref)
        if not isinstance(score, list):
            score = [score]
        for s in score:
            if math.isfinite(s) and not math.isnan(s):
                self.updated += 1
                self.scores.append(s)

    def get_num_scores(self):
        return len(self.scores)

    def get_all_scores(self):
        return self.scores

    def get_last_score(self):
        return self.scores[-1]

    def get_last_scores(self, n):
        return self.scores[-n:]

    def get_mean_score(self):
        if self.get_num_scores() == 0:
            return -1
        mean_score = sum(self.scores) / self.get_num_scores()
        return mean_score

    def get_name(self):
        return self.name


class FWLMetric(BaseMetric):
    def __init__(self):
        super().__init__(name='fwl')
        self.event_images = None

    def calculate(self, evs, flow, output_images=False, reverse_time=False):
        output = voxel_warping_flow_loss(evs, flow, output_images=output_images, reverse_time=reverse_time)
        FWL0 = voxel_warping_flow_loss(evs, torch.zeros_like(flow))
        
        if isinstance(output, tuple):
            self.event_images = output[1]
            score = output[0] / FWL0
        else:
            score = output / FWL0
        return score
    
    def update(self, evs, flow, output_images=False, reverse_time=False):
        assert evs is not None and flow is not None
        self.updated = 0
        score = self.calculate(evs, flow, output_images, reverse_time)
        if not isinstance(score, list):
            score = [score]
        for s in score:
            if math.isfinite(s) and not math.isnan(s):
                self.updated += 1
                self.scores.append(s)


class PsnrMetric(BaseMetric):

    def __init__(self):
        super().__init__(name='psnr')

    def calculate(self, img, ref):
        score = psnr(ref, img)
        return score


class MseMetric(BaseMetric):

    def __init__(self):
        super().__init__(name='mse')

    def calculate(self, img, ref):
        score = mse(ref, img)
        return score


class SsimMetric(BaseMetric):

    def __init__(self, gaussian_weights=True, sigma=1.5, use_sample_covariance=False):
        super().__init__(name='ssim')
        self.gaussian_weights = gaussian_weights
        self.sigma = sigma
        self.use_sample_covariance = use_sample_covariance

    def calculate(self, img, ref):
        score = ssim(ref, img, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        return score


class PyIqaMetricFactory:
    """Factory class to create PyIQA metrics"""

    def __init__(self):
        import pyiqa
        self.pyiqa = pyiqa
        self.list_of_metrics = pyiqa.list_models()
        self.batch_size = 4
        self.created_metrics = {}

    def get_metric(self, name):
        if name in self.created_metrics:
            return self.created_metrics[name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            iqa_metric = self.pyiqa.create_metric(name)
        no_ref = True if iqa_metric.metric_mode == 'NR' else False
        metric_name = name.lower()

        def inference(self):
            if len(self.image_queue) < 1:
                return []
            img_tensor = torch.cat(self.image_queue[-self.batch_size:])
            if self.ref_queue[0] is not None:
                ref_tensor = torch.cat(self.ref_queue[-self.batch_size:])
                score_tensor = iqa_metric(img_tensor, ref_tensor)
            else:
                score_tensor = iqa_metric(img_tensor)
            self.image_queue = []
            self.ref_queue = []
            score_list = score_tensor.squeeze().tolist()
            if not isinstance(score_list, list):
                score_list = [score_list]
            return score_list

        def finish_queue(self):
            self.updated = 0
            score = self.inference()
            self.updated += len(score)
            self.scores.extend(score)

        def calculate(self, img, ref=None):
            img_tensor = cv2torch(img, num_ch=3)
            ref_tensor = None if ref is None else cv2torch(ref, num_ch=3)
            self.image_queue.append(img_tensor)
            self.ref_queue.append(ref_tensor)
            if len(self.image_queue) < self.batch_size:
                return []
            return self.inference()

        pyiqa_metric_class = type('PyIqa_' + name + '_Metric', (BaseMetric,), {'calculate': calculate,
                                                                               'inference': inference,
                                                                               'finish_queue': finish_queue})
        metric_obj = pyiqa_metric_class(metric_name, no_ref)
        self.created_metrics[name] = metric_obj

        return metric_obj


pyiqa_metric_factory = PyIqaMetricFactory()


class EvalMetricsTracker:
    """
    Helper class to calculate and keep track of all the qualitative results and quantitative evaluation metrics.
    Performs the following tasks:
    - Calculates the requested quantitative evaluation metrics within the specified time window.
    - Keeps track of the calculated quantitative evaluation metrics.
    - Saves qualitative results to filesystem (if specified).
    - Histogram equalization of images (if specified).
    - Saves processed images to filesystem (if specified).
    - Generating videos from images (when create_video is called).
    """

    def __init__(self, save_images=False, save_processed_images=False, save_events=False, save_interval=1, output_dir=None, hist_eq='none',
                 quan_eval_metric_names=None, quan_eval_start_time=0, quan_eval_end_time=float('inf'),
                 quan_eval_ts_tol_ms=float('inf'), has_reference_frames=False, color=False):
        if quan_eval_metric_names is None:
            quan_eval_metric_names = ['mse', 'psnr', 'ssim', 'lpips'] #'fwl'
        self.save_images = save_images
        self.save_events = save_events
        # self.save_flow = save_flow
        self.save_processed_images = save_processed_images
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.output_flow_dir = join(output_dir, 'flow')#------ output folder for flow
        
        self.hist_eq = hist_eq
        self.quan_eval_start_time = quan_eval_start_time
        self.quan_eval_end_time = quan_eval_end_time
        self.quan_eval_ts_tol_ms = quan_eval_ts_tol_ms
        self.has_reference_frames = has_reference_frames
        self.color = color
        self.quan_eval_indices = []

        if self.hist_eq == 'none' and self.save_processed_images:
            print("Can not save processed images when hist_eq is none")
            self.save_processed_images = False

        self.metrics = []
        for metric_name in quan_eval_metric_names:
            if metric_name == "mse":
                self.metrics.append(MseMetric())
            elif metric_name == "ssim":
                self.metrics.append(SsimMetric())
            elif metric_name == "psnr": #----
                self.metrics.append(PsnrMetric())
            elif metric_name == "fwl": #-----
                self.metrics.append(FWLMetric())
            elif metric_name in pyiqa_metric_factory.list_of_metrics:
                self.metrics.append(pyiqa_metric_factory.get_metric(metric_name))
            else:
                print("Unknown metric " + metric_name)

        if not self.has_reference_frames:
            self.metrics = [m for m in self.metrics if m.no_ref]

        self.only_no_ref = all([m.no_ref for m in self.metrics])

        self.reset()

    def reset(self):
        self.setup_output_folders_and_files()
        for metric in self.metrics:
            metric.reset()

    def save_new_scores(self, idx, metric):
        metric_file_path = self.get_metric_file_path(metric)
        num_updated = metric.get_num_updated()
        if num_updated > 0:
            last_scores = metric.get_last_scores(num_updated)
            indices = self.quan_eval_indices[-num_updated:]
            append_result(metric_file_path, indices, last_scores)

    def finalize(self, idx):
        for metric in self.metrics:
            metric.finish_queue()
            self.save_new_scores(idx, metric)

    def update_quantitative_metrics(self, idx, img, ref, evs=None, flow=None):
        self.quan_eval_indices.append(idx)
        event_images = None
        for metric in self.metrics:
            try:
                #------ add fwl
                if metric.name == 'fwl':
                    metric.update(evs, flow, output_images=True)
                    event_images = metric.event_images #voxel_grid and voxel_grid_warped
                elif not self.has_reference_frames or metric.no_ref:
                    metric.update(img)
                else:
                    metric.update(img, ref)
                self.save_new_scores(idx, metric)
            except Exception as e:
                print("Exception in metric " + metric.get_name() + ": " + str(e))
                print(traceback.format_exc())
                metric.reset()
        return event_images
    
    def update(self, idx, img, ref, img_ts, ref_ts=None, flow=None, evs=None):
        if ref_ts is None:
            ref_ts = img_ts

        # prepare timestamps file
        timestamps_file_name = self.get_timestamps_file_path()
        append_timestamp(timestamps_file_name, idx, img_ts)

        # clip images
        img = np.clip(img, 0.0, 1.0)
        if self.has_reference_frames:
            ref = np.clip(ref, 0.0, 1.0)
        #------------
        org_img = np.copy(img)
        img = self.histogram_equalization(img)
        if self.has_reference_frames:
            ref = self.histogram_equalization(ref)

        inside_eval_cut = self.quan_eval_start_time <= img_ts <= self.quan_eval_end_time
        img_ref_time_diff_ms = abs(ref_ts - img_ts) * 1000
        inside_eval_ts_tolerance = img_ref_time_diff_ms <= self.quan_eval_ts_tol_ms
        if inside_eval_cut and inside_eval_ts_tolerance:
            event_images = self.update_quantitative_metrics(idx, img, ref, evs, flow)
            #------------
            if idx ==0 or (idx%self.save_interval) == 0:
                if self.save_images:
                    save_inferred_image(self.output_dir, org_img, idx)
                if self.save_processed_images:
                    save_inferred_image(self.processed_output_dir, img, idx)
                if self.save_images and flow is not None:
                    ensure_dir(self.output_flow_dir)
                    flow = torch2cv2(flow)
                    rgb_flow = merge_optical_flow(flow)
                    save_flow(self.output_flow_dir, rgb_flow, idx)
                if self.save_events:
                    save_events(self.output_evs_dir, event_images, idx)



    def save_custom_metric(self, idx, metric_name, metric_value, is_int=False):
        metric_file_path = join(self.output_dir, metric_name + '.txt')
        if idx == 0:
            open(metric_file_path, 'w', encoding="utf-8").close()  # overwrite with emptiness
        append_result(metric_file_path, idx, metric_value, is_int)

    def get_num_quan_evaluations(self):
        return len(self.quan_eval_indices)

    def create_video(self):
        if self.save_images:
            create_vid_from_recon_folder(self.output_dir)
        else:
            print("Can not create video when save_images is False")

    def create_processed_video(self):
        if self.save_processed_images:
            ts_path = self.get_timestamps_file_path()
            shutil.copy2(ts_path, self.processed_output_dir)
            create_vid_from_recon_folder(self.processed_output_dir)
        else:
            print("Can not create processed video when save_processed_images is False")

    def get_mean_scores(self):
        mean_scores = {}
        for metric in self.metrics:
            name = metric.get_name()
            mean_score = metric.get_mean_score()
            mean_scores[name] = mean_score
        return mean_scores

    def get_timestamps_file_path(self):
        timestamps_path = join(self.output_dir, 'timestamps.txt')
        return timestamps_path

    def get_metric_file_path(self, metric):
        metric_name = metric.get_name()
        metric_file_path = join(self.output_dir, metric_name + '.txt')
        return metric_file_path

    def setup_output_folders_and_files(self):
        ensure_dir(self.output_dir)
        if self.save_processed_images:
            self.processed_output_dir = self.output_dir + "_processed"
            ensure_dir(self.processed_output_dir)
        if self.save_events:
            self.output_evs_dir = join(self.output_dir, 'events') #self.output_dir + "events"
            ensure_dir(self.output_evs_dir)

        timestamps_file_name = self.get_timestamps_file_path()
        open(timestamps_file_name, 'w', encoding="utf-8").close()  # overwrite with emptiness
        for metric in self.metrics:
            metric_file_path = self.get_metric_file_path(metric)
            open(metric_file_path, 'w', encoding="utf-8").close()  # overwrite with emptiness

    def histogram_equalization(self, img):
        if self.hist_eq == 'global':
            from skimage.util import img_as_ubyte, img_as_float32
            from skimage import exposure
            img = exposure.equalize_hist(img)
            img = img_as_float32(img)
        elif self.hist_eq == 'local':
            from skimage.morphology import disk
            from skimage.filters import rank
            from skimage.util import img_as_ubyte, img_as_float32
            footprint = disk(55)
            img = img_as_ubyte(img)
            img = rank.equalize(img, footprint=footprint)
            img = img_as_float32(img)
        elif self.hist_eq == 'clahe':
            from skimage.util import img_as_ubyte, img_as_float32
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = img_as_ubyte(img)
            img = clahe.apply(img)
            img = img_as_float32(img)
        elif self.hist_eq == 'none':
            pass
        else:
            raise ValueError(f"Unrecognized histogram equalization argument: {self.hist_eq}")
        return img
