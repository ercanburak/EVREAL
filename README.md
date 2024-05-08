# CISTA-EVREAL 

This repository is the modification of [EVREAL](https://github.com/ercanburak/EVREAL) (Event-based Video Reconstruction Evaluation and Analysis Library) by intergrating our family of CISTA networks, which includes [CISTA-TC](https://ieeexplore.ieee.org/abstract/document/9746331), [CISTA-LSTC](https://ieeexplore.ieee.org/abstract/document/10130595) and [CISTA-Flow](https://arxiv.org/pdf/2403.11961). The original repository for CISTA-TC and CISTA-LSTC is [V2E2V](https://github.com/lsying009/V2E2V). Furthermore, we have made some changes for our evaluation, as described in [CISTA-Flow](https://arxiv.org/pdf/2403.11961). The main modifications are listed below:

## Modifications

### Intergrate CISTA family and our pretrained models

We added code for our CISTA family networks in ```CISTAFlow/e2v/e2v_model.py```, including CISTA, CISTA-TC, CISTA-LSTC and CISTA-Flow. The ```DCEIFlowCistaNet``` and ```ERAFTCistaNet``` are both CISTA-Flow networks that integrate [DCEIFlow](https://github.com/danqu130/DCEIFlow) and [ERAFT](https://github.com/uzh-rpg/E-RAFT) into CISTA-LSTC for motion compensation, respectively. In the evaluation, these two networks are referred to as CISTA-EIFlow and CISTA-ERAFT.

The pretrained models are located under ```pretrained```. Note that CISTA-LSTC has two checkpoints: ```cista-lstc-old.pth.tar``` and ```cista-lstc.pth.tar```. The ```cista-lstc.pth.tar``` checkpoint is used in the [CISTA-Flow paper](https://arxiv.org/pdf/2403.11961), while the old version is used in the [TPAMI paper](https://ieeexplore.ieee.org/abstract/document/10130595) and the [V2E2V repository](https://github.com/lsying009/V2E2V). The training datasets and strategies are slightly different for these two checkpoints. Note that the new version is not necessarily better than the old one.

In addition, we have included pretrained models for E2VID, FireNet, and SPADE-E2VID, labeled as ```myE2VID```, ```myFireNet``` and ```mySPADE```.


### New evaluation method: ```low_k_events```

We have added a new evaluation method: ```low_k_events```. If the target number of events per reconstruction is $k$, We either divide events between two consecutive frames by $k$ into multiple groups or combine events across several frames into a single group for the reconstruction and flow estimation. Related configuration is in ```config/eval/low_k_events.json```.

```json
"voxel_method": {
            "method": "low_k_events",
            "k": 15000
        },
```

### New evaluation metrics
We have added PSNR and [forward warping loss (FWL)](https://link.springer.com/chapter/10.1007/978-3-030-58583-9_32) in evaluation. The FWL is only used for optical flow estimation of CISTA-Flow when no ground truth flow is available.

### Other modifications
```save_interval: int```: save data (frame / events / flow) every ```save_interval``` frames

```save_events: bool``` : is save corresponding event frames


## Usage
For detailed usage, please refer to [EVREAL](https://github.com/ercanburak/EVREAL). 

Example:
```bash
python eval.py \
-m CISTA-TC CISTA-LSTC CISTA-EIFlow CISTA-ERAFT \
-c low_k_events \
-d ECD HQF \
-qm mse psnr ssim lpips \
```

## Acknowledgements

Related open-source repositories:
- [V2E2V](https://github.com/lsying009/V2E2V) 
- [DCEIFlow](https://github.com/danqu130/DCEIFlow)
- [ERAFT](https://github.com/uzh-rpg/E-RAFT)
- [EVREAL](https://github.com/ercanburak/EVREAL) 
- [event_cnn_minimal](https://github.com/TimoStoff/event_cnn_minimal)

We also provided codes for generating training datasets for video-to-events reconstruction:
- [V2E_generation](https://github.com/lsying009/V2E_Generation)

## Citations
If you use this library, please cite the following:
```bibtex
  @inproceedings{ercan2023evreal,
    title={{EVREAL}: Towards a Comprehensive Benchmark and Analysis Suite for Event-based Video Reconstruction},
    author={Ercan, Burak and Eker, Onur and Erdem, Aykut and Erdem, Erkut},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month={June},
    year={2023},
    pages={3942-3951}}
```

If you use the family of CISTA networks, please cite the publications as follows:
```bibtex
  @misc{liu2024enhanced,
    title={Enhanced Event-Based Video Reconstruction with Motion Compensation}, 
    author={Siying Liu and Pier Luigi Dragotti},
    year={2024},
    eprint={2403.11961},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
```bibtex
  @article{liu_sensing_2023,  
    title={Sensing Diversity and Sparsity Models for Event Generation and Video Reconstruction from Events},   
    author={Liu, Siying and Dragotti, Pier Luigi},  
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
    year={2023},  
    pages={1-16},  
    publisher={IEEE}. 
    doi={10.1109/TPAMI.2023.3278940}. 
    }
```
```bibtex
  @inproceedings{liu_convolutional_2022,  
    title={Convolutional ISTA Network with Temporal Consistency Constraints for Video Reconstruction from Event Cameras},  
    author={Liu, Siying and Alexandru, Roxana and Dragotti, Pier Luigi},  
    booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
    pages={1935--1939},  
    year={2022},  
    organization={IEEE}. 
    doi={10.1109/ICASSP43922.2022.9746331}. 
    }
```
