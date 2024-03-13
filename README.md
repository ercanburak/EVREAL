# EVREAL (Event-based Video Reconstruction Evaluation and Analysis Library)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://ercanburak-evreal.hf.space/)
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2305.00434-B31B1B.svg)](https://arxiv.org/abs/2305.00434/)

This is the official repository for the [CVPRW 2023](https://tub-rip.github.io/eventvision2023/) paper **[EVREAL: Towards a Comprehensive Benchmark and Analysis Suite for Event-based Video Reconstruction](https://arxiv.org/abs/2305.00434)** by [Burak Ercan](https://ercanburak.github.io/), [Onur Eker](https://github.com/ekeronur/), [Aykut Erdem](https://aykuterdem.github.io/), and [Erkut Erdem](https://web.cs.hacettepe.edu.tr/~erkut/).

In this work, we present **EVREAL**, an open-source framework that offers a unified evaluation pipeline to comprehensively benchmark PyTorch based **pre-trained neural networks** for **event-based video reconstruction**, and a result analysis tool to **visualize and compare** reconstructions and their scores.

![An overall look at our proposed EVREAL (Event-based Video Reconstruction – Evaluation and Analysis Library) toolkit.](https://ercanburak.github.io/projects/evreal/diagram.png "An overall look at our proposed EVREAL (Event-based Video Reconstruction – Evaluation and Analysis Library) toolkit.")


For more details please see our [paper](https://arxiv.org/abs/2305.00434). 

For qualitative and quantitative experimental analyses please see our [project website](https://ercanburak.github.io/evreal.html), or [interactive demo](https://ercanburak-evreal.hf.space/).

## News
- Codes for color reconstruction added.
- Evaluation codes are published now. Please see below for [installation](#installation), [dataset preparation](#preparing-datasets) and [usage](#usage) instructions. (Codes for robustness analysis and downstream tasks will be published soon.)
- In our [result analysis tool](https://ercanburak-evreal.hf.space/), we also share results of **a new state-of-the-art model, HyperE2VID,** which generates higher-quality videos than previous state-of-the-art, while also reducing memory consumption and inference time. Please see the [HyperE2VID webpage](https://ercanburak.github.io/HyperE2VID.html) for more details.
- The web application of our result analysis tool is ready now. [Try it here](https://ercanburak-evreal.hf.space/) to interactively visualize and compare qualitative and quantitative results of event-based video reconstruction methods.
- We will present our work at the CVPR Workshop on Event-Based Vision in person, on the 19th of June 2023, during Session 2 (starting at 10:30 local time). Please see the [workshop website](https://tub-rip.github.io/eventvision2023/) for details.

## Installation

Installing the packages required for evaluation, in a conda environment named `evreal`:
```
conda create -y -n evreal python=3.10
conda activate evreal
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Preparing Datasets

EVREAL processes event datasets in a numpy memmap format similar to the format of [event_utils](https://github.com/TimoStoff/event_utils/) library. The datasets used for the experiments presented in our paper are the [ECD](https://rpg.ifi.uzh.ch/davis_data.html), [MVSEC](https://daniilidis-group.github.io/mvsec/), [HQF](https://timostoff.github.io/20ecnn), [BS-ERGB](https://github.com/uzh-rpg/timelens-pp/), and the [HDR](https://rpg.ifi.uzh.ch/E2VID.html) datasets.

In the [tools](tools) folder, we present tools to download these datasets and convert their format. Here, you can find commands to install a separate conda environment named `evreal-tools`, and commands to download and install aforementioned datasets (except BS-ERGB). After preparing the datasets with these commands, the dataset folder should look as follows:

```
├── EVREAL
  ├── data
    ├── ECD  # First dataset
      ├── boxes_6dof  # First sequence
        ├── events_p.npy
        ├── events_ts.npy
        ├── events_xy.npy
        ├── image_event_indices.npy
        ├── images.npy
        ├── images_ts.npy
        ├── metadata.json
      ├── calibration  # Second sequence
      ├── ...
    ├── HQF  # Second dataset
    ├── ...
```

## Usage

The main script of EVREAL is [eval.py](eval.py). This script takes a set of evaluation settings, methods, datasets, and evaluation metrics as command line arguments, and generates reconstructed images and scores using each given method and for each sequence in the given datasets. For example, the following command evaluates the E2VID and HyperE2VID **m**ethods using the `std` evaluation **c**onfiguration, on the ECD **d**ataset, using the MSE, SSIM and LPIPS **q**uantitative **m**etrics:

```bash
python eval.py -m E2VID HyperE2VID -c std -d ECD -qm mse ssim lpips
```

Each evaluation configuration, method, and dataset has a specific json formatted file in [config/eval](config/eval), [config/method](config/method), and [config/dataset](config/dataset) folders, respectively. Therefore the example command given above reads the settings to use from the [std.json](config/eval/std.json), [ECD.json](config/dataset/ECD.json), [E2VID.json](config/method/E2VID.json), and [HyperE2VID.json](config/method/HyperE2VID.json) files. For the MSE and SSIM metrics, the implementations from [scikit-image library](https://scikit-image.org/docs/stable/api/skimage.metrics.html) are used. For all the other metrics, we use the [PyTorch IQA Toolbox](https://github.com/chaofengc/IQA-PyTorch/).

#### Outputs

After the evaluation is finished, the scores of each method in each dataset are printed in a table format, and the results are stored in `outputs` folder in the following structure:
```
├── EVREAL
  ├── outputs
    ├── std  # Evaluation config
      ├── ECD  # Dataset
        ├── boxes_6dof  # Sequence 
          ├── E2VID  # Method
            ├── mse.txt  # MSE scores
            ├── ssim.txt  # SSIM scores
            ├── lpips.txt  # LPIPS scores
            ├── frame_0000000000.png  # Reconstructed frames 
            ├── frame_0000000001.png
            ├── ...
          ├── HyperE2VID
            ├── ...
```

#### Example Commands

To generate full-reference metric scores for each method on four datasets (as in Table 2 of the paper, with the addition of the new method HyperE2VID):

```bash
python eval.py -m E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID -c std -d ECD MVSEC HQF BS_ERGB_handheld -qm mse ssim lpips
```

To generate no-reference metric scores for each method on ECD-FAST and MVSEC-NIGHT datasets (as in Table 3 of the paper, with the addition of the new method HyperE2VID):
```bash
python eval.py -m E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID -c std -d ECD_fast MVSEC_night -qm brisque niqe maniqa
```

To generate no-reference metric scores for each method on HDR datasets (as in Table 3 of the paper, with the addition of the new method HyperE2VID):
```bash
python eval.py -m E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID -c t40ms -d TPAMI20_HDR -qm brisque niqe maniqa
```

To generate color reconstructions on CED dataset:
```bash
python eval.py -m HyperE2VID -c color -d CED 
```

## Citations

If you use this library in an academic context, please cite the following:

```
@inproceedings{ercan2023evreal,
title={{EVREAL}: Towards a Comprehensive Benchmark and Analysis Suite for Event-based Video Reconstruction},
author={Ercan, Burak and Eker, Onur and Erdem, Aykut and Erdem, Erkut},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month={June},
year={2023},
pages={3942-3951}}
```

## Acknowledgements

- This work was supported in part by KUIS AI Center Research Award, TUBITAK-1001 Program Award No. 121E454, and BAGEP 2021 Award of the Science Academy to A. Erdem.
- This code borrows from or inspired by the following open source repositories:
  - https://github.com/TimoStoff/event_cnn_minimal
  - https://github.com/TimoStoff/event_utils/
- Here are the open-source repositories (with model codes and pretrained models) of the methods that we compare with EVREAL:
  - [E2VID](https://github.com/uzh-rpg/rpg_e2vid)
  - [FireNet](https://github.com/cedric-scheerlinck/rpg_e2vid/tree/cedric/firenet)
  - [E2VID+](https://github.com/TimoStoff/event_cnn_minimal)
  - [FireNet+](https://github.com/TimoStoff/event_cnn_minimal)
  - [SPADE-E2VID](https://github.com/RodrigoGantier/SPADE_E2VID)
  - [SSL-E2VID](https://github.com/tudelft/ssl_e2vid)
  - [ET-Net](https://github.com/WarranWeng/ET-Net)
  - [HyperE2VID](https://github.com/ercanburak/HyperE2VID)
