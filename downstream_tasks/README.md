## EVREAL Downstream Tasks


### Image Classification
```bash
conda activate evreal-tools
gdown 1Gr8bxBSV2hwqPF8SkkT7fK9YzZSje9U8 -O downstream_tasks/classification/
conda activate evreal
python eval.py -m E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID -c t60ms_s -d NCaltech101 
python tools/organize_NCaltech101_recons.py 
python downstream_tasks/classification/classifier.py 
```
