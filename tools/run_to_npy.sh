#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=200gb:ngpus=0
#PBS -lwalltime=01:30:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate evreal-tools

# python tools/bag_to_npy.py /rds/general/user/sl220/home/data/EVREAL_data/ECD --remove

python tools/bag_to_npy.py /rds/general/user/sl220/home/data/EVREAL_data/MVSEC --event_topic /davis/left/events --image_topic /davis/left/image_raw --remove