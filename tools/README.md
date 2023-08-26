## EVREAL Tools for Downloading and Processing Datasets

Installing the packages required for downloading and processing datasets, in a separate conda environment named `evreal-tools`
```bash
conda create -n evreal-tools python=3.8
conda activate evreal-tools
pip install -r tools/requirements.txt
```

Commands to download and convert the format of all datasets (run from the root directory of EVREAL):
```bash
./tools/download_ECD.sh
python tools/bag_to_npy.py data/ECD --remove
./tools/download_MVSEC.sh
python tools/bag_to_npy.py data/MVSEC --event_topic /davis/left/events --image_topic /davis/left/image_raw --remove
./tools/download_HQF.sh
python tools/bag_to_npy.py data/HQF --remove
./tools/download_TPAMI20_HDR.sh
python tools/txt_to_npy.py data/TPAMI20/ --flip --remove
```
