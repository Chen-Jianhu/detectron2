#!/bin/bash -e
# This script is used for download ILSVRC2015 DET and ILSVRC2015 VID dataset.

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
if [[ ${CWD} != */datasets ]]; then
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

# # Download datasets
# echo "Start downloading the dataset ..."
# # 1. ILSVRC2015 DET
# echo "Download LSVRC2015 DET datasets (3 subset) ..."
# BASE_URL="http://image-net.org/image/ILSVRC2015/"
# for dataset in ILSVRC2015_DET.tar.gz ILSVRC2015_DET_test.tar.gz ILSVRC2015_DET_test_new.tar.gz; do
#     dest=${ROOT}/${dataset}
#     echo "Dwonlaod ${dataset} to ${dest} ..."
#     curl ${BASE_URL}/${dataset} --output ${dest}
# done
# # 2. ILSVRC2015 VID
# echo "Download LSVRC2015 VID datasets ..."
# curl http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz --output ${ROOT}

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
for image_index in DET_train_30classes.txt VID_train_15frames.txt VID_train_every10frames.txt VID_val_frames.txt VID_val_videos.txt; do
    dest=${ROOT}/ILSVRC2015/ImageSets/${image_index}
    echo "Dwonlaod ${image_index} to ${dest} ..."
    wget ${BASE_URL}/${image_index} -O ${dest}
done

# Add soft links
D2_DATASETS_ROOT=$(cd "$(dirname "$0")";pwd)
echo "Add soft links: ${ROOT}/ILSVRC2015 => ${D2_DATASETS_ROOT}/ILSVRC2015"
ln -s ${ROOT}/ILSVRC2015 ${D2_DATASETS_ROOT}/ILSVRC2015

echo "Complete!"

