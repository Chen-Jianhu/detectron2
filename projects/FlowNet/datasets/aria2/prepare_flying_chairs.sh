#!/bin/bash -e
# This script is used for download Flying Chairs dataset.
# See: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
# for more details.
# This script will use the aria2 tool to speed up,
# see: https://aria2.github.io/ for more details.
# Note: You should install aria2 before run this srcipt.
# For example:
#   ```bash
#   sudo apt update
#   sudo apt install aria2
#   ````

function help () {
    echo "This script is used for download Flying Chairs dataset."
    echo ""
    echo "Usage: "
    echo "  ./prepare_for_flying_chairs.sh [download_dir]"
    echo ""
    echo "Example: "
    echo "  ./prepare_for_flying_chairs.sh /data/datasets/"
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

aria2c https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip \
--dir ${ROOT} -j 50

# Unzip datasets
echo "Start to unzip the dataset ..."
cd ${ROOT}
unzip FlyingChairs.zip
cd -

# Add soft links
D2_DATASETS_ROOT=$(cd ..;pwd)
echo "Add soft links: ${ROOT}/FlyingChairs_release => ${D2_DATASETS_ROOT}/FlyingChairs_release"
ln -s ${ROOT}/FlyingChairs_release ${D2_DATASETS_ROOT}/

echo "Complete!"

