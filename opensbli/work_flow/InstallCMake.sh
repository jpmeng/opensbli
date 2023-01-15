#!/bin/bash
##@brief Download and install a specific version of CMake
##@author Jianping Meng
##@contributors
##@details

if [ $# -ne 3 ]
then
    echo "This script will download, compile and install CMake to a specified directory!"
    echo "Usage: ./InstallCMake.sh Dir Version User"
    echo "Dir: the directory to install CMake!"
    echo "Version: the CMake version"
    echo "Version number might be 3.23.0 3.21.0 3.20.0 3.19.0 ..."
    echo "User (Y/N): if the installation is in user space! If not, we need to sudo!"
    exit 1
fi
cmake_dir=$1
version=$2
user=$3
wget -c https://github.com/Kitware/CMake/releases/download/v$version/cmake-$version-Linux-x86_64.sh
# sudo is not necessary for directories in user space.
mkdir $cmake_dir
sh ./cmake-$version-Linux-x86_64.sh --prefix=$cmake_dir  --skip-license
if [ $user == "N" ]
then
    if [ ! -d "/usr/local/bin" ]
    then
        mkdir -p /usr/local/bin
    fi
    ln -s $cmake_dir/bin/cmake /usr/local/bin/cmake
fi

if [ $user=="Y" ]
then
    if [ ! -d "$HOME/bin" ]
    then
        mkdir -p $HOME/bin
    fi
    ln -s $cmake_dir/bin/cmake $HOME/bin/cmake
    echo "Please add $HOME/bin to path for enabling CMake"
fi
rm cmake-$version-Linux-x86_64.sh