import argparse
import glob
import os
import traceback
from collections import OrderedDict

import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm
from yachalk import chalk

import model as model_arch
from dataset import MemMapDataset
from utils.eval_metrics import EvalMetricsTracker
from utils.eval_utils import torch2cv2, normalize
from utils.timers import CudaTimer
from utils.util import CropParameters, read_json, get_height_width

color_progress = chalk.cyan.bold
color_error = chalk.red.bold
color_scores = chalk.green.bold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_eval_configs(eval_config_names):
    eval_configs = []
    for eval_config_name in eval_config_names:
        eval_config_path = os.path.join("config", "eval", eval_config_name + ".json")
        eval_config = read_json(eval_config_path)
        eval_config['name'] = eval_config_name
        eval_configs.append(eval_config)
    return eval_configs


def get_sequences(dataset_config, dataset_kwargs):
    """
    Get a list of sequences for a dataset config. Each sequence is a dict with the following keys:
    - name: name of the sequence
    - data_loader: a PyTorch dataLoader object for the sequence
    - start_time_s: start time of quantitative evaluation for the sequence in seconds
    - end_time_s: end time of quantitative evaluation for the sequence in seconds
    """
    dataset_root = dataset_config['root_path']
    get_all_sequences = dataset_config.get('get_all_sequences', False)
    has_subfolders = dataset_config.get('has_subfolders', False)
    dataset_kwargs.update(dataset_config.get('dataset_kwargs', {}))
    sequences = []

    if get_all_sequences:
        if has_subfolders:
            glob_pattern = os.path.join(dataset_root, '*', '*')
        else:
            glob_pattern = os.path.join(dataset_root, '*')
        sequence_paths = glob.glob(glob_pattern)
        sequences_config = OrderedDict()
        for sequence_path in sequence_paths:
            if has_subfolders:
                sequence_name = os.path.basename(os.path.dirname(sequence_path)) + "_" + os.path.basename(sequence_path)
            else:
                sequence_name = os.path.basename(sequence_path)
            sequences_config[sequence_name] = {'sequence_path': sequence_path}
    else:
        sequences_config = dataset_config.get('sequences', {})

    for sequence_name, sequence in tqdm(sequences_config.items()):
        sequence_path = sequence.get('sequence_path', os.path.join(dataset_root, sequence_name))
        sequence['name'] = sequence_name
        dataset = MemMapDataset(sequence_path, **dataset_kwargs)
        sequence['data_loader'] = DataLoader(dataset, pin_memory=True)
        min_t, max_t = dataset.get_min_max_t()
        if 'start_time_s' not in sequence:
            sequence['start_time_s'] = min_t
        if 'end_time_s' not in sequence:
            sequence['end_time_s'] = max_t
        sequences.append(sequence)
    return sequences


def get_dataset_configs(dataset_names):
    dataset_configs = []
    for dataset_name in dataset_names:
        dataset_config_path = os.path.join("config", "dataset", dataset_name + ".json")
        dataset_config = read_json(dataset_config_path)
        dataset_config['name'] = dataset_name
        dataset_configs.append(dataset_config)
    return dataset_configs


def get_datasets(dataset_configs, dataset_kwargs):
    datasets = []
    for dataset_config in dataset_configs:
        dataset = {'name': dataset_config['name']}
        sequences = get_sequences(dataset_config, dataset_kwargs)
        dataset['sequences'] = sequences
        datasets.append(dataset)
    return datasets


def get_number_of_sequences_in_all_datasets(datasets):
    sequence_count = 0
    for dataset in datasets:
        sequence_count += len(dataset['sequences'])
    return sequence_count


def load_model(model, state_dict):
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_method_config(method_name):
    method_config_path = os.path.join("config", "method", method_name + ".json")
    method_config = read_json(method_config_path)
    return method_config


def get_model_from_checkpoint_path(model_name, checkpoint_path):
    """
    Instantiate a PyTorch model class from a given model name and checkpoint path. According to the model name,
    each checkpoint is parsed and each model class is instantiated with the correct parameters.
    """
    checkpoint = torch.load(checkpoint_path, device)
    if model_name == "SPADE-E2VID":
        model = model_arch.SpadeE2vid()
        model.num_encoders = 3
        state_dict = checkpoint
    elif model_name == "SSL-E2VID":
        unet_kwargs = {"base_num_channels": 32, "kernel_size": 5, "num_bins": 5, "num_encoders": 3,
                       "recurrent_block_type": "convlstm", "num_residual_blocks": 2, "skip_type": "sum", "norm": None,
                       "use_upsample_conv": True}
        model = model_arch.E2VIDRecurrent(unet_kwargs)
        state_dict = checkpoint
    else:
        if model_name == "E2VID":
            unet_kwargs = checkpoint['model']
            unet_kwargs['final_activation'] = 'sigmoid'
            model = model_arch.E2VIDRecurrent(unet_kwargs)
        elif model_name == "FireNet":
            unet_kwargs = checkpoint['config']['model']
            unet_kwargs['final_activation'] = ''
            model = model_arch.FireNet_legacy(unet_kwargs)
        else:
            config = checkpoint['config']
            model = config.init_obj('arch', model_arch)
            if model_name == "ET-Net":
                model.num_encoders = 3
            elif model_name == "FireNet+":
                model.num_encoders = 0
        state_dict = checkpoint['state_dict']
    model = load_model(model, state_dict)
    return model


def get_cropper(model, data_loader):
    height, width = get_height_width(data_loader)
    cropper = CropParameters(width, height, model.num_encoders)
    return cropper


def get_eval_metrics_tracker(dataset_name, eval_config, method_name, sequence, metrics):
    output_path = os.path.join("outputs", eval_config['name'], dataset_name, sequence['name'], method_name)

    save_images = eval_config.get('save_images', True)
    save_processed_images = save_images and eval_config['histeq'] != 'none'

    has_reference_frames = sequence["data_loader"].dataset.has_images

    eval_metrics_tracker = EvalMetricsTracker(save_images=save_images,
                                              save_processed_images=save_processed_images,
                                              output_dir=output_path,
                                              hist_eq=eval_config['histeq'],
                                              quan_eval_metric_names=metrics,
                                              quan_eval_start_time=sequence['start_time_s'],
                                              quan_eval_end_time=sequence['end_time_s'],
                                              quan_eval_ts_tol_ms=eval_config['ts_tol_ms'],
                                              has_reference_frames=has_reference_frames)
    return eval_metrics_tracker


def eval_method_on_sequence(dataset_name, eval_config, method_name, model, method_config, sequence, metrics):
    """
    Evaluates a method on a single sequence from a dataset, using the given evaluation configuration and metrics.
    """
    eval_metrics_tracker = get_eval_metrics_tracker(dataset_name, eval_config, method_name, sequence, metrics)
    data_loader = sequence['data_loader']
    has_reference_frames = data_loader.dataset.has_images
    cropper = get_cropper(model, data_loader)
    model.reset_states()
    eval_infer_all = eval_config.get('eval_infer_all', False)
    post_process_norm = method_config.get('post_process_norm', "none")
    event_tensor_normalization = method_config.get('event_tensor_normalization', False)
    idx = 0
    for idx, item in enumerate(tqdm(data_loader)):
        if has_reference_frames:
            ref_frame = item['frame']
            ref_frame_ts = item['frame_timestamp'].item()
        else:
            ref_frame = None
            ref_frame_ts = None
        pred_frame_ts = item['voxel_timestamp'].item()
        # Only start reconstruction when close to eval start (10 seconds)
        if pred_frame_ts < sequence['start_time_s'] - 10 and not eval_infer_all:
            continue
        if pred_frame_ts > sequence['end_time_s'] and not eval_infer_all:
            idx -= 1
            break
        if item['event_count'].item() <= 1 or item['dt'].item() == 0:
            event_rate = 0
        else:
            event_rate = item['event_count'].item() / item['dt'].item()
        voxel = item['events']
        if event_tensor_normalization:
            voxel = normalize_event_tensor(voxel)
        voxel = voxel.to(device)
        voxel = cropper.pad(voxel)
        with CudaTimer(method_name):
            output = model(voxel)
        image = cropper.crop(output['image'])
        image = torch2cv2(image)
        image = post_process_normalization(image, post_process_norm)
        if has_reference_frames:
            ref_frame = torch2cv2(ref_frame)
        eval_metrics_tracker.update(idx, image, ref_frame, pred_frame_ts, ref_frame_ts)
        eval_metrics_tracker.save_custom_metric(idx, "event_rate", event_rate)
    eval_metrics_tracker.finalize(idx)
    num_evaluated = eval_metrics_tracker.get_num_quan_evaluations()
    mean_scores = eval_metrics_tracker.get_mean_scores()
    if eval_config['create_video']:
        eval_metrics_tracker.create_video()
        if eval_config['histeq'] != 'none':
            eval_metrics_tracker.create_processed_video()
    return num_evaluated, mean_scores


class MetricTracker:
    def __init__(self):
        self.data_dict = {}

    def init_key(self, key):
        self.data_dict[key] = {}
        self.data_dict[key]['total'] = 0.0
        self.data_dict[key]['count'] = 0
        self.data_dict[key]['average'] = 0.0

    def update(self, key, value, count=1):
        if key not in self.data_dict:
            self.init_key(key)
        self.data_dict[key]['total'] += value * count
        self.data_dict[key]['count'] += count
        self.data_dict[key]['average'] = self.data_dict[key]['total'] / self.data_dict[key]['count']

    def get_average(self, key):
        if key not in self.data_dict:
            self.init_key(key)
        return self.data_dict[key]['average']


def print_scores(all_metrics, method_names, dataset_names, config_name):
    """
    Prints all the scores for an eval config, in a table format.
    """
    scores_table = []
    headers = ["\nMethod"]
    for method_name, method_metrics in zip(method_names, all_metrics):
        weighted_averages = []
        for dataset_name, dataset_metrics in zip(dataset_names, method_metrics):
            for idx, metric in enumerate(dataset_metrics.data_dict):
                if idx == 0:
                    headers.append(dataset_name + "\n" + metric.upper())
                else:
                    headers.append("\n" + metric.upper())
                weighted_average_score = dataset_metrics.get_average(metric)
                weighted_averages.append(weighted_average_score)
        scores_table.append([method_name] + weighted_averages)
        # weighted_averages = (['{:.5f}'.format(x) for x in weighted_averages])
        # weighted_averages = ','.join(weighted_averages)
        # print(scores_color(weighted_averages))
    print('')
    print(chalk.underline(color_scores(f'Image Quality Scores (for {config_name} config)')))
    print(color_scores(tabulate(scores_table, headers=headers, floatfmt=".3f")))
    print('')


def get_eval_info_str(eval_config, method_names, dataset_configs):
    """
    Returns a string with the evaluation information, for logging purposes.
    """
    if len(method_names) > 1:
        methods_str = "methods " + method_names[0]
        for method_name in method_names[1:-1]:
            methods_str = methods_str + ", " + method_name
        methods_str += " and " + method_names[-1]
    else:
        methods_str = "method " + method_names[0]

    if len(dataset_configs) > 1:
        datasets_str = dataset_configs[0]['name']
        for dataset in dataset_configs[1:-1]:
            datasets_str = datasets_str + ", " + dataset['name']
        datasets_str = datasets_str + " and " + dataset_configs[-1]['name']
        datasets_str += " datasets"
    else:
        datasets_str = dataset_configs[0]['name'] + " dataset"

    eval_config_name = eval_config['name']
    eval_info_str = ("evaluating " + methods_str + " on " + datasets_str + " with " +
                     eval_config_name + " evaluation config")
    return eval_info_str


def eval_method_with_config(eval_config, method_name, datasets, metrics):
    """
    Evaluates a method with a given evaluation config, on the given datasets and using the given metrics.
    """
    num_sequences = get_number_of_sequences_in_all_datasets(datasets)
    method_config = get_method_config(method_name)
    print(color_progress("Starting method " + method_name))
    checkpoint_path = method_config['model_path']
    model_name = method_config['model_name']
    method_metrics = []
    try:
        model = get_model_from_checkpoint_path(model_name, checkpoint_path)
    except Exception as e:
        print(color_error(f"Exception while getting method {method_name} from checkpoint path {checkpoint_path}"))
        print(color_error(e))
        print(color_error(traceback.format_exc()))
        return method_metrics

    sequence_no = 1
    for dataset in datasets:
        dataset_metrics = None
        try:
            sequences = dataset['sequences']
            dataset_metrics = MetricTracker()
            for sequence in sequences:
                print(color_progress(f"Evaluating {method_name} method with {eval_config['name']} evaluation config"
                                     f" on {sequence['name']} sequence from {dataset['name']} dataset. "
                                     f"({sequence_no}/{num_sequences} for this method and config)"))
                results = eval_method_on_sequence(dataset['name'], eval_config, method_name, model, method_config, sequence, metrics)
                num_evaluated, mean_scores = results
                sequence_no += 1
                for metric_name, score in mean_scores.items():
                    dataset_metrics.update(metric_name, score, num_evaluated)
        except Exception as e:
            print(color_error(f"Exception while evaluating method {method_name} on {dataset['name']} dataset:"))
            print(color_error(e))
            print(color_error(traceback.format_exc()))
        finally:
            if dataset_metrics:
                method_metrics.append(dataset_metrics)

    return method_metrics


def post_process_normalization(img, norm):
    """
    Post-process an image with standard or robust normalization.
    """
    if norm == 'robust':
        img = normalize(img, 1, 99)
    elif norm == 'standard':
        img = normalize(img, 0, 100)
    elif norm == 'none':
        pass
    elif norm == 'exprobust':
        img = np.exp(img)
        img = normalize(img, 1, 99)
    else:
        raise ValueError(f"Unrecognized normalization argument: {norm}")
    return img


def normalize_event_tensor(event_tensor):
    """
    Normalize event tensors such that the mean and stddev of the nonzero values in each tensor is 0 and 1, respectively.
    """
    nonzero = event_tensor != 0
    num_nonzeros = nonzero.sum()
    if num_nonzeros > 0:
        mean = event_tensor.sum() / num_nonzeros
        stddev = torch.sqrt((event_tensor ** 2).sum() / num_nonzeros - mean ** 2)
        stddev = torch.max(stddev, torch.tensor(1e-6))
        mask = nonzero.float()
        event_tensor = mask * (event_tensor - mean) / stddev
    return event_tensor


def evaluate(method_names, eval_config_names=None, dataset_names=None, metrics=None):
    """
    Evaluate the given methods on the given datasets with the given evaluation configs and metrics.
    :param method_names: list of method names to evaluate. Each has a corresponding config file in config/method.
        Available methods are E2VID, E2VID+, FireNet, FireNet+, SPADE-E2VID, SSL-E2VID, ET-Net, and HyperE2VID.
    :param eval_config_names: list of evaluation config names to use (from config/eval).
    :param dataset_names: list of dataset config names to use (from config/dataset).
    :param metrics: list of metrics to use. Available metrics are MSE, SSIM, and any other metric from
        the https://github.com/chaofengc/IQA-PyTorch repo.
    """
    if method_names is None:
        method_names = ['E2VID', 'E2VID+', 'FireNet', 'FireNet+', 'SPADE-E2VID', 'SSL-E2VID', 'ET-Net', 'HyperE2VID']
    if eval_config_names is None:
        eval_config_names = ['std']
    if dataset_names is None:
        dataset_names = ['ECD', 'MVSEC', 'HQF']
    if metrics is None:
        metrics = ['mse', 'ssim', 'lpips']
    eval_configs = get_eval_configs(eval_config_names)
    dataset_configs = get_dataset_configs(dataset_names)
    for eval_config in eval_configs:
        dataset_kwargs = eval_config.get('dataset_kwargs', {})
        datasets = get_datasets(dataset_configs, dataset_kwargs)
        eval_info_str = get_eval_info_str(eval_config, method_names, dataset_configs)
        print(color_progress("Started " + eval_info_str))
        config_all_metrics = []
        for method_name in method_names:
            method_metrics = eval_method_with_config(eval_config, method_name, datasets, metrics)
            config_all_metrics.append(method_metrics)
        print(color_progress("Finished " + eval_info_str))
        dataset_names = [dataset['name'] for dataset in datasets]
        print_scores(config_all_metrics, method_names, dataset_names, eval_config['name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='event2im evaluation script')
    parser.add_argument('-c', '--config', nargs='+', type=str, help='evaluation configs')
    parser.add_argument('-m', '--method', nargs='+', type=str, help='methods')
    parser.add_argument('-d', '--dataset', nargs='+', type=str, help='datasets')
    parser.add_argument('-qm', '--metrics', nargs='+', type=str,
                        help='quantitative evaluation metrics that will be used calculate scores')
    args = parser.parse_args()
    evaluate(args.method, args.config, args.dataset, args.metrics)
