#!/bin/bash -e
# This script is used for download ILSVRC2015 DET and ILSVRC2015 VID dataset.
# This script will use the aria2 tool to speed up,
# see: https://aria2.github.io/ for more details.
# Note: You should install aria2 before run this srcipt.
# For example:
#   ```bash
#   sudo apt update
#   sudo apt install aria2
#   ````

function help () {
    echo "This script is used for download ILSVRC2015 DET and ILSVRC2015 VID dataset."
    echo ""
    echo "Usage: "
    echo "  ./prepare_for_vid.sh [download_dir]"
    echo ""
    echo "Example: "
    echo "  ./prepare_for_vid.sh /data/datasets/"
}

CWD=$(pwd)
echo "CWD: ${CWD}"
if [[ ${CWD} != */aria2 ]]; then
    SCRIPT_DIR=$(cd "$(dirname "$0")";pwd)
    echo ""
    echo "Error: You should run this script in path: ${SCRIPT_DIR}."
    echo ""
    help
    exit
fi

if [ $# -eq 0 ]; then
    ROOT="/data/datasets"
    echo "Download datasets to the default data directory: ${ROOT}"
    mkdir -p ${ROOT}
elif [ $# -eq 1 ]; then
    ROOT=$1
    echo "Download datasets to the specified directory: ${ROOT}"
    mkdir -p ${ROOT}
else
    help
    exit
fi

# Download datasets
echo "Start downloading the dataset ..."
DET_BASE_URL="http://image-net.org/image/ILSVRC2015"
VID_BASE_URL="http://bvisionweb1.cs.unc.edu/ilsvrc2015"

aria2c -Z -P ${DET_BASE_URL}/{ILSVRC2015_DET.tar.gz,ILSVRC2015_DET_test.tar.gz,ILSVRC2015_DET_test_new.tar.gz} \
${VID_BASE_URL}/ILSVRC2015_VID.tar.gz \
--dir ${ROOT} -j 50

# Unzip datasets
echo "Start to unzip the dataset ..."
cd ${ROOT}
tar -zxvf ILSVRC2015_DET.tar.gz
tar -zxvf ILSVRC2015_DET_test.tar.gz
tar -zxvf ILSVRC2015_DET_test_new.tar.gz
tar -zxvf ILSVRC2015_VID.tar.gz
cd -

# Download image index
BASE_URL="https://raw.githubusercontent.com/JianhuChen/mega.pytorch/master/datasets/ILSVRC2015/ImageSets/"
aria2c -Z -P ${BASE_URL}/{DET_train_30classes.txt,VID_train_15frames.txt,VID_train_every10frames.txt,VID_val_frames.txt,VID_val_videos.txt} \
--dir ${ROOT}/ILSVRC2015/ImageSets/

# Add soft links
D2_DATASETS_ROOT=$(cd ..;pwd)
echo "Add soft links: ${ROOT}/ILSVRC2015 => ${D2_DATASETS_ROOT}/ILSVRC2015"
ln -s ${ROOT}/ILSVRC2015 ${D2_DATASETS_ROOT}/ILSVRC2015

echo "Complete!"

