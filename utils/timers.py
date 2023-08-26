import atexit
from collections import defaultdict

import numpy as np
import torch
from yachalk import chalk

cuda_timers = defaultdict(list)


class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))


def print_timing_info():
    colored = chalk.yellow.bold
    print(chalk.underline(colored('== Timing statistics ==')))
    for timer_name, timing_values in cuda_timers.items():
        timing_value = np.mean(np.array(timing_values))
        print(colored(f'{timer_name}: {timing_value:.2f} ms ({len(timing_values)} samples)'))


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
