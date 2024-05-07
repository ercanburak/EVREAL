#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=16gb:ngpus=1
#PBS -lwalltime=00:20:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate py311

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"

#mse psnr ssim lpips brisque niqe
# myE2VID120 myFireNet120 mySPADE120 CISTA-LSTC120  CISTA-ERAFT CISTA
# CISTA CISTA-D CISTA-TC CISTA-LSTC-d1 CISTA-LSTC-d3 CISTA-LSTC-d5 CISTA-LSTC-d7 CISTA-LSTM CISTA-LSTC-Z0 CISTA-LSTC-old \
# low_k_events/ std for SIM
# python eval.py \
-m CISTA-TC20 \
-c low_k_events \
-d ECD HQF \
-qm mse psnr ssim lpips \

python eval.py \
-m CISTA-TC20 \
-c std \
-d SIM \
-qm mse psnr ssim lpips \



# save every 10
# python eval.py \
-m CISTA-TC \
-c low_k_events \
-d ECD_fast MVSEC_night \
-qm brisque \


# python eval.py \
-m E2VID+ FireNet+ SPADE-E2VID ET-Net HyperE2VID CISTA CISTA-D CISTA-TC CISTA-LSTC CISTA-EIFlow \
-c low_k_events \
-d ECD \
-qm mse \

# python eval.py \
-m CISTA-ERAFT \
-c low_k_events \
-d ECD_fast \
-qm brisque niqe \

# CISTA-ERAFT 

# python eval.py \
-m CISTA-EIFlow CISTA-ERAFT \
-c low_k_events \
-d MVSEC_night MVSEC \
-qm brisque niqe \


# python eval.py \
-m E2VID+ SPADE-E2VID FireNet+ \
-c low_k_events \
-d ECD_fast MVSEC_night \
-qm brisque niqe \

# E2VID+ SPADE-E2VID FireNet+
# python eval.py \
-m E2VID+ myE2VID FireNet+ myFireNet SPADE-E2VID mySPADE ET-Net HyperE2VID CISTA-LSTC CISTA-EIFlow CISTA-ERAFT \
-c low_k_events \
-d ECD HQF \
-qm mse psnr ssim lpips brisque niqe

# python eval.py \
-m E2VID+ myE2VID FireNet+ myFireNet SPADE-E2VID mySPADE ET-Net HyperE2VID CISTA-LSTC CISTA-EIFlow CISTA-ERAFT \
-c low_k_events \
-d MVSEC_night \
-qm brisque niqe

# python eval.py \
-m E2VID+ myE2VID FireNet+ myFireNet SPADE-E2VID mySPADE ET-Net HyperE2VID CISTA-LSTC CISTA-EIFlow CISTA-ERAFT \
-c low_k_events \
-d ECD_fast \
-qm brisque niqe \
