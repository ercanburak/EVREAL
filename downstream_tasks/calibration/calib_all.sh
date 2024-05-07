sudo echo "Starting calibration"
for iterno in {1..10}; do
  for modelName in E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID groundtruth; do
    CALIBDIR=calibdir\_$modelName/iter$iterno
    python images_to_rosbag.py --rosbag_folder $CALIBDIR --image_folder ../../outputs/std_all/ECD_calib/calibration/$modelName --image_topic /dvs/image_reconstructed
    cp target.yaml run_calib.sh $CALIBDIR
    xhost local:root
    sudo docker run -e DISPLAY=$DISPLAY  --net=host -v "$(pwd)/$CALIBDIR:/calib" mzahana/kalibr:latest bash calib/run_calib.sh
  done 
done