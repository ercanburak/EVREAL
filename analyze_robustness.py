import os
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd


def order_models(models):
    """
    Orders the models based on the models_order list.
    """
    models_to_ignore = []
    models_order = ['E2VID', 'FireNet', 'E2VID+', 'FireNet+', 'SPADE-E2VID', 'SSL-E2VID', 'ET-Net', 'HyperE2VID']
    models_order.reverse()
    models_ordered = sorted(models, key=lambda x: models_order.index(x) if x in models_order else 99999)
    return [model for model in models_ordered if model not in models_to_ignore]


def extract_numeric_value(s):
    """
    Extracts numeric value from a string like 't100ms' or 'k10'.
    Returns a float or int found in the string, handling common cases.
    """
    # Use regular expression to find numeric values in the string
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"Cannot extract numeric value from {s}")


def read_event_sparsity_and_lpips(directory):
    """
    Reads LPIPS scores and event sparsity values, aligning and aggregating them for plotting.
    """
    data = []

    datasets_list = ["ECD", "MVSEC", "HQF"]
    for dataset_name in datasets_list:
        dataset_path = os.path.join(directory, dataset_name)
        sequences_list = glob.glob(dataset_path + '/*')
        for sequence_path in sequences_list:
            models_list = glob.glob(sequence_path + '/*')
            for model_path in models_list:
                model_name = os.path.basename(model_path)
                event_sparsity_path = os.path.join(model_path, 'event_rate.txt')
                lpips_path = os.path.join(model_path, 'lpips.txt')

                if os.path.exists(lpips_path) and os.path.exists(event_sparsity_path):
                    lpips_scores = []
                    lpips_indices = []
                    with open(lpips_path, 'r') as f:
                        for line in f.readlines():
                            lpips_scores.append(float(line.split(' ')[-1]))
                            lpips_indices.append(int(line.split(' ')[0]))

                    event_sparsity_values = []
                    event_sparsity_indices = []
                    with open(event_sparsity_path, 'r') as f:
                        for line in f.readlines():
                            event_sparsity_values.append(float(line.split(' ')[-1]))
                            event_sparsity_indices.append(int(line.split(' ')[0]))

                    if len(lpips_scores) == len(event_sparsity_values):
                        for es, lpips in zip(event_sparsity_values, lpips_scores):
                            data.append({'model': model_name, 'event_sparsity': es, 'lpips': lpips})
                    else:
                        # If the number of scores is different, align the indices
                        lpips_dict = dict(zip(lpips_indices, lpips_scores))
                        event_sparsity_dict = dict(zip(event_sparsity_indices, event_sparsity_values))
                        common_indices = set(lpips_indices) & set(event_sparsity_indices)
                        for idx in common_indices:
                            data.append({'model': model_name, 'event_sparsity': event_sparsity_dict[idx],
                                         'lpips': lpips_dict[idx]})

    return pd.DataFrame(data)


def read_lpips_scores(directory, pattern='*'):
    """
    Reads LPIPS scores from text files in a nested directory structure.
    Returns a dictionary with model names as keys and another dictionary
    mapping conditions to LPIPS score lists.
    """
    scores_dict = {}
    lpips_min, lpips_max = 1, 0
    dirs_list = sorted(glob.glob(os.path.join(directory, pattern)))

    for path in dirs_list:
        datasets_list = glob.glob(path + '/*')
        condition = os.path.basename(path)
        for dataset_path in datasets_list:
            sequences_list = glob.glob(dataset_path + '/*')
            for sequence_path in sequences_list:
                models_list = glob.glob(sequence_path + '/*')
                for model_path in models_list:
                    if not os.path.isdir(model_path):
                        continue
                    model_name = os.path.basename(model_path)
                    if model_name in ['HyperE2VID']:
                        continue
                    scores_dict.setdefault(model_name, {}).setdefault(condition, [])
                    lpips_path = os.path.join(model_path, 'lpips.txt')
                    if not os.path.exists(lpips_path):
                        continue
                    with open(os.path.join(model_path, 'lpips.txt'), 'r') as f:
                        scores = [float(line.split(' ')[-1]) for line in f.readlines()]
                        scores_dict[model_name][condition].extend(scores)

    # Calculate mean scores and update min/max
    for model_scores in scores_dict.values():
        for condition, scores in model_scores.items():
            mean_score = np.mean(scores)
            model_scores[condition] = mean_score
            lpips_min = min(lpips_min, mean_score)
            lpips_max = max(lpips_max, mean_score)

    return scores_dict, lpips_min, lpips_max


def plot_results(scores_dict, lpips_min, lpips_max, xlabel, ylabel, transform_condition=lambda x: int(x)):
    """
    Plots the results from the LPIPS scores dictionary.
    """
    markers = ['o', 'd', '8', '*', 'v', 'X', 's', 'p', 'P', 'h', 'H', 'D', '1', '2', '3', '4', 'x', '|', '_', '8', 's',
               'p', '*', 'h', 'H', 'd', 'D', 'v', '^', '<', '>']
    condition_list, lpips_list = [], []  # Initialize as empty lists

    models_ordered = order_models(list(scores_dict.keys()))

    for idx, model_name in enumerate(models_ordered):
        condition_dict = scores_dict[model_name]
        sorted_conditions = sorted((transform_condition(cond), lpips) for cond, lpips in condition_dict.items())
        if sorted_conditions:  # Check if sorted_conditions is not empty
            conditions, lpips_scores = zip(*sorted_conditions)  # Unzip into two lists
            condition_list.extend(conditions)  # Extend the master condition_list
            plt.plot(conditions, lpips_scores, linestyle='--', marker=markers[idx % len(markers)], label=model_name)

    if condition_list:
        # plt.xticks(sorted(set(condition_list)), fontsize=20)
        plt.xticks(sorted(set(condition_list)))
        # plt.yticks(fontsize=20)
        plt.ylim(lpips_min - 0.02, lpips_max + 0.02)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Position legend in the upper right
        # plt.xlabel(xlabel, fontsize=24)
        # plt.ylabel(ylabel, fontsize=24)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        # plt.savefig("a" + ".pdf", format="pdf", dpi=600, bbox_inches="tight")
        plt.show()
    else:
        print("No data available for plotting.")


def process_directory(directory, pattern, xlabel, ylabel, condition_transformer=extract_numeric_value):
    """
    Processes a directory to read LPIPS scores and plot the results.
    """
    scores_dict, lpips_min, lpips_max = read_lpips_scores(directory, pattern=pattern)
    # lpips_min = 0.3
    # lpips_max = 0.7
    plot_results(scores_dict, lpips_min, lpips_max, xlabel, ylabel, condition_transformer)


class OOMFormatter(ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def plot_event_sparsity_results(df, xlabel="Event Sparsity", ylabel="LPIPS"):
    """
    Plots LPIPS scores aggregated into buckets based on event sparsity values.
    """
    markers = ['o', 'd', '8', '*', 'v', 'X', 's', 'p', 'P', 'h', 'H', 'D', '1', '2', '3', '4', 'x', '|', '_', '8', 's',
               'p', '*', 'h', 'H', 'd', 'D', 'v', '^', '<', '>']
    pretrained_names = list(df['model'].unique())
    pretrained_names = order_models(list(pretrained_names))
    lpips_min, lpips_max = 1, 0
    for idx, model_name in enumerate(pretrained_names):
        model_df = df[df['model'] == model_name]
        buckets = pd.cut(model_df['event_sparsity'], 10, include_lowest=True)
        buckets_lpips_es = model_df['lpips'].groupby(buckets).mean()
        if buckets_lpips_es.min() < lpips_min:
            lpips_min = buckets_lpips_es.min()
        if buckets_lpips_es.max() > lpips_max:
            lpips_max = buckets_lpips_es.max()
        idx_list = [bucket.mid for bucket in buckets_lpips_es.index.categories]
        plt.plot(idx_list, buckets_lpips_es.values, linestyle='--', marker=markers[idx % len(markers)],
                 label=model_name)
    lpips_min, lpips_max = 0.3, 0.7
    plt.ylim(lpips_min - 0.02, lpips_max + 0.02)
    # plt.xticks(idx_list)
    # plt.xlabel(xlabel, fontsize=24)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))
    # plt.gca().xaxis.get_offset_text().set_fontsize(18)
    # plt.xticks(idx_list, fontsize=20)
    plt.xticks(idx_list)
    # plt.yticks(fontsize=20)
    # plt.legend(loc='upper right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Position legend in the upper right
    plt.tight_layout()
    # plt.savefig("b" + ".pdf", format="pdf", dpi=600, bbox_inches="tight")
    plt.show()


def process_event_sparsity(directory, xlabel="Event Sparsity", ylabel="LPIPS"):
    """
    Processes the directory for event sparsity scores and plots the LPIPS results.
    """
    event_sparsity_df = read_event_sparsity_and_lpips(directory)
    plot_event_sparsity_results(event_sparsity_df, xlabel=xlabel, ylabel=ylabel)


if __name__ == "__main__":
    base_dir = 'outputs'
    process_directory(base_dir, "t*ms", "duration (ms)", "LPIPS")
    process_directory(base_dir, "k*k", "# of events in groups [K]", "LPIPS")
    process_directory(base_dir, "kr*", "ratio of discarded frames", "LPIPS", lambda x: 1.0 if x == 'std' else 1 - float(x[2:]))
    process_event_sparsity(os.path.join(base_dir, "std"), "event rate (events/sec.)", "LPIPS")
