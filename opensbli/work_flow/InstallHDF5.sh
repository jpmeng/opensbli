#!/bin/bash
##@brief Download, compile and install HDF5 library
##@author Jianping Meng
##@contributors
##@details

function usage {
    echo "This script will download, compile and install the HDF library to a specified directory."
    echo "./$(basename $0) -h -> showing usage"
    echo "./$(basename $0) -v -> Specifying the HDF5 version (e.g., 1.10.1)"
    echo "./$(basename $0) -d -> Specifying the directory where the HDF5 to be installed."
    echo "./$(basename $0) -f -> Enabling the Fortran support."
    echo "./$(basename $0) -s -> Compiling in the sequential mode (note OPS requires the parallel by default if using CMake)."
}
optstring=":v:dfsh"
HDF5Ver="1.12.1"
Dir="$HOME/HDF5"
Parallel="ON"
Fortran="OFF"

while getopts ${optstring} options; do
    case ${options} in
        h)
            usage
            exit 0
        ;;
        v)
            HDF5Ver=${OPTARG}
        ;;
        s)
            Parallel="OFF"
        ;;
        f)
            Fortran="ON"
        ;;
        d)
            Dir=${OPTARG}
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
    echo "We will install parallel hdf5-${HDF5Ver} to ${Dir} without Fortran support."
fi
if [ ${Parallel} == "ON" ]
then
    echo "Depending on the system, we might need to set CC=mpicc to compile in parallel mode."
    echo "Typically in a desktop we need to run the script as CC=mpicc ./InstallHDF5.sh"
    echo "However, in ARCHER2, we just need to run ./InstallHDF5.sh"
    sleep 5
fi


MainVer="${HDF5Ver:0:$((${#HDF5Ver}-2))}"
wget -c "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$MainVer/hdf5-$HDF5Ver/src/CMake-hdf5-$HDF5Ver.tar.gz"
cpus=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
if [ ${cpus} -gt 12 ]
then
    cpus=12
fi
tar -xf "CMake-hdf5-$HDF5Ver.tar.gz"
cd CMake-hdf5-$HDF5Ver/hdf5-${HDF5Ver}
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${Dir} -DCMAKE_BUILD_TYPE=Release -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_ENABLE_PARALLEL=${Parallel} -DHDF5_BUILD_FORTRAN=${Fortran}
cmake --build . --config Release -j ${cpus}
cmake --install . --config Release
cd ../../../
rm -r -f "CMake-hdf5-$HDF5Ver"
rm -r -f "CMake-hdf5-$HDF5Ver.tar.gz"