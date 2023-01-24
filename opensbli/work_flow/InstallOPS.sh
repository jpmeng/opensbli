#!/bin/bash
##@brief Download, compile and install OPS library
##@author Jianping Meng
##@contributors Pushpender Sharma Teja Ala
##@details

function usage {
    echo "This script will download, compile and install the OPS library to a specified directory!"
    echo "./$(basename $0) -h -> Showing usage"
    echo "./$(basename $0) -d -> Specifying the directory for installation"
    echo "./$(basename $0) -c -> Specifying the compiler"
    echo "./$(basename $0) -H -> Specifying the HDF5 directory"
    echo "./$(basename $0) -o -> Specifying the branch (e.g.,feature/HDF5Slice)"
    echo "./$(basename $0) -m -> Specifying the machine type"
    echo "Machine type can be: Ubuntu (default) ARCHER2 IRIDIS5 Fedora"
}
optstring="hc:m:d:H:o:"
Compiler="Gnu"
Dir="$HOME/OPS_INSTALL"
Machine="Ubuntu"
HDF5Root=""
Branch="develop"

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
        H)
            HDF5Root="-DHDF5_ROOT=${OPTARG}"
        ;;
        o)
            Branch="${OPTARG}"
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

if [ $# -eq 0 ]
then
    echo "This script will download, compile with ${Compiler} and install the OPS library to to ${Dir}!"
fi

if ! grep "CreateOpenSBLIEnv" /proc/$PPID/cmdline
then
    if [ $Machine == "Ubuntu" ]
    then
        HDF5Lib="libhdf5-openmpi-dev"
        if [ -z ${HDF5Root} ]
        then
            HDF5Lib=""
        fi
        sudo apt install libhdf5-mpi-dev build-essential ${HDF5Lib}
    fi
    if [ $Machine == "Fedora" ]
    then
        HDF5Lib="hdf5-openmpi-devel hdf5-devel"
        if [ -z ${HDF5Root} ]
        then
            HDF5Lib=""
        fi
        sudo dnf install make automake gcc gcc-c++ kernel-devel ${HDF5Lib}
    fi
fi

if [ $Machine == "ARCHER2" ]
then
    module purge PrgEnv-cray
    module load load-epcc-module
    module load cmake/3.21.3
    module load PrgEnv-gnu
    module load cray-hdf5-parallel
fi


if [ $Machine == "IRIDIS5" ]
then
    module load gcc/6.4.0
    module load hdf5/1.10.2/gcc/parallel
    module load cuda/10.0
    module load cmake
fi

#https://github.com/OP-DSL/OPS/archive/refs/heads/feature/HDF5Slice.zip
#https://github.com/OP-DSL/OPS/archive/refs/heads/BurgerEquation.zip
#OPS-feature-HDF5Slice/
#OPS-BurgerEquation

wget -c https://github.com/OP-DSL/OPS/archive/refs/heads/${Branch}.zip
FileName="$(basename -- $Branch)"
unzip "${FileName}.zip"
rm -r -f "${FileName}.zip"
SourceDir=OPS-`echo ${Branch} | sed  's/\//-/g'`
cd ${SourceDir}
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${Dir} -DCMAKE_BUILD_TYPE=Release -DCFLAG="-ftree-vectorize -funroll-loops" -DCXXFLAG="-ftree-vectorize -funroll-loops" -DBUILD_OPS_APPS=OFF ${HDF5Root}
cmake --build . -j 4
cmake --install .
cd ../../
rm -r -f ${SourceDir}