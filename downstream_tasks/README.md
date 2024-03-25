## EVREAL Downstream Tasks

### Object Detection
```bash
conda activate evreal
python eval.py -m E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID -c std -d MVSEC_night_1
python tools/extract_gt_images.py data/MVSEC/outdoor_night1_data/ outputs/std/MVSEC_night_1/outdoor_night1_data/groundtruth
conda activate evreal-tools
cd downstream_tasks/detection
bash detect_all.sh
```

### Image Classification
```bash
conda activate evreal-tools
gdown 1Gr8bxBSV2hwqPF8SkkT7fK9YzZSje9U8 -O downstream_tasks/classification/
conda activate evreal
python eval.py -m E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID -c t60ms_s -d NCaltech101 
python tools/organize_NCaltech101_recons.py 
python downstream_tasks/classification/classifier.py 
```
