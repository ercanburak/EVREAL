# EVREAL (Event-based Video Reconstruction Evaluation and Analysis Library)

This is the official repository for the [CVPRW 2023](https://tub-rip.github.io/eventvision2023/) paper **[EVREAL: Towards a Comprehensive Benchmark and Analysis Suite for Event-based Video Reconstruction](https://arxiv.org/abs/2305.00434)** by [Burak Ercan](https://ercanburak.github.io/), [Onur Eker](https://github.com/ekeronur/), [Aykut Erdem](https://aykuterdem.github.io/), and [Erkut Erdem](https://web.cs.hacettepe.edu.tr/~erkut/).

In this work, we present **EVREAL**, an open-source framework that offers a unified evaluation pipeline to comprehensively benchmark PyTorch based **pre-trained neural networks** for **event-based video reconstruction**, and a result analysis tool to **visualize and compare** reconstructions and their scores.

![An overall look at our proposed EVREAL (Event-based Video Reconstruction – Evaluation and Analysis Library) toolkit.](https://ercanburak.github.io/projects/evreal/diagram.png "An overall look at our proposed EVREAL (Event-based Video Reconstruction – Evaluation and Analysis Library) toolkit.")


For more details please see our [paper](https://arxiv.org/abs/2305.00434). 

For qualitative and quantitative experimental analyses please see our [project website](https://ercanburak.github.io/evreal.html), or [interactive demo](https://ercanburak-evreal.hf.space/).

Codes will be published soon.

## News

- In our [result analysis tool](https://ercanburak-evreal.hf.space/), we also share results of a new event-based video reconstruction model, HyperE2VID, which generates higher-quality videos than previous state-of-the-art, while also reducing memory consumption and inference time. Please see the [HyperE2VID webpage](https://ercanburak.github.io/HyperE2VID.html) for more details.
- The web application of our result analysis tool is ready now. [Try it here](https://ercanburak-evreal.hf.space/) to interactively visualize and compare qualitative and quantitative results of event-based video reconstruction methods.
- We will present our work at the CVPR Workshop on Event-Based Vision in person, on the 19th of June 2023, during Session 2 (starting at 10:30 local time). Please see the [workshop website](https://tub-rip.github.io/eventvision2023/) for details.

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

This work was supported in part by KUIS AI Center Research Award, TUBITAK-1001 Program Award No. 121E454, and BAGEP 2021 Award of the Science Academy to A. Erdem.
