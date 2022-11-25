##@brief Download, compile and install OPS library
##@author Jianping Meng
##@contributors
##@details
#!/bin/bash
if [ $# -eq 0 ]
then
    echo "This script will download, compile and install the OPS library to a specified directory!"
    echo "Usage: ./InstallOPS.sh Dir  MachineType"
    echo "Dir: the directory to install OPS ($HOME/OPS_INSTALL by default!)"
    echo "MachineType (Ubuntu ARCHER2) Ubuntu by default"
    exit 1
fi
dir=$HOME/OPS_INSTALL
if [ $# -eq 1 ]
then
    dir=$1
fi
machine="Ubuntu"
if [ $# -eq 2 ]
then
    machine=$2
fi

if [ $machine == "ARCHER2" ]
then
    module purge PrgEnv-cray
    module load load-epcc-module
    module load cmake/3.21.3
    module load PrgEnv-gnu
    module load cray-hdf5-parallel
fi

if [ $machine == "Ubuntu" ]
then
    sudo apt install libhdf5-openmpi-dev libhdf5-mpi-dev build-essential
fi

wget -c https://github.com/OP-DSL/OPS/archive/refs/heads/develop.zip
unzip develop.zip
rm develop.zip
cd OPS-develop
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=$1 -DCMAKE_BUILD_TYPE=Release -DCFLAG="-ftree-vectorize -funroll-loops"
-DCXXFLAG="-ftree-vectorize -funroll-loops" -DBUILD_OPS_APPS=OFF
cmake --build . -j 4
cmake --install .
cd ../../
rm -r -f OPS-develop
