#!/bin/bash
##@brief Create  a working environment for OpenSBLI!
##@author Jianping Meng
##@contributors
##@details

function usage {
    echo "This script will download, compile and install the OPS library to a specified directory!"
    echo "./$(basename $0) -h -> Showing usage"
    echo "./$(basename $0) -b -> Specifying OpenSBLI branch (e.g.,cpc_release)"
    echo "./$(basename $0) -o -> Specifying OPS branch (e.g.,feature/HDF5Slice) "
    echo "./$(basename $0) -d -> Specifying the directory (absolute path) for creating the environment"
    echo "./$(basename $0) -p -> Use Python2"
    echo "./$(basename $0) -H -> if installing a local HDF5 library"
    echo "./$(basename $0) -c -> Specifying the compiler"
    echo "./$(basename $0) -m -> Specifying the machine type"
    echo "Machine type can be: Ubuntu (default) ARCHER2 IRIDIS5 Fedora"
}
#./InstallHDF5.sh
optstring="hb:o:d:p:HC:m:"
Compiler="Gnu"
Branch="cpc_release"
Dir="$HOME/OpenSBLI"
Machine="Ubuntu"
LocalHDF5="OFF"
PythonVer="3"
OpsBranch="develop"

while getopts ${optstring} options; do
    case ${options} in
        h)
            usage
            exit 0
        ;;
        c)
            Compiler=${OPTARG}
        ;;
        m)
            Machine=${OPTARG}
        ;;
        d)
            Dir=${OPTARG}
        ;;
        b)
            Branch=${OPTARG}
        ;;
        o)
            OpsBranch=${OPTARG}
        ;;
        H)
            LocalHDF5="ON"
        ;;
        p)
            PythonVer="3"
        ;;
        :)
            echo "$0: Must supply an argument to -$OPTARG." >&2
            exit 1
        ;;
        ?)
            echo "Invalid option: -${OPTARG}."
            exit 2
        ;;
    esac
done
## Install Gnu compiler provided by system
if [ $Machine == "Ubuntu" ]
then
    HDF5Lib="libhdf5-openmpi-dev"
    if [[ "${LocalHDF5}"  == "ON" ]]
    then
        HDF5Lib=""
    fi
    sudo apt install libhdf5-openmpi-dev  build-essential
fi
if [ $Machine == "Fedora" ]
then
    HDF5Lib="hdf5-openmpi-devel hdf5-devel"
    if [[ "${LocalHDF5}"  == "ON" ]]
    then
        HDF5Lib=""
    fi
    sudo dnf install make automake gcc gcc-c++ kernel-devel ${HDF5Lib}
fi

if [ -d "$Dir" ]
then
   echo "The specified directory is exising, please delete it first or change to another one!"
   exit 1
fi
mkdir -p $Dir
#cd $Dir ## needed by normal version
## Download the scripts

# wget -c https://github.com/jpmeng/utilities/archive/refs/heads/main.zip
# unzip -j main.zip
# rm main.zip
## Install local HDF5 if needed
WithHDF5=""
if [[ "${LocalHDF5}"  == "ON" ]]
then
    if [ -d "HDF5" ]
    then
        rm -r -f HDF5
    fi
    mkdir ${Dir}/HDF5
    WithHDF5="-H ${Dir}/HDF5"
    if [[ "${Machine}" == "Ubuntu" ]] ||  [[ "${Machine}" == "Fedora" ]]
    then
        export CC=mpicc
    fi
    ./InstallHDF5.sh -d "${Dir}/HDF5"
fi
# Python
./InstallPython.sh -d $Dir/Python
# OPS
./InstallOPS.sh -d $Dir/OPS-INSTALL -m ${Machine} ${WithHDF5} -b ${OpsBranch}
# OpenSBLI
wget -c https://github.com/opensbli/opensbli/archive/refs/heads/${Branch}.zip
FileName="$(basename -- $Branch)"
unzip "${FileName}.zip"
rm -r -f "${FileName}.zip"
DefaultOpenSBLIDir=opensbli-`echo ${Branch} | sed  's/\//-/g'`
mv ${DefaultOpenSBLIDir} ${Dir}/OpenSBLI
#wget -c https://github.com/opensbli/opensbli/archive/refs/heads/cpc_release.zip
#unzip cpc_release.zip
#rm cpc_release.zip