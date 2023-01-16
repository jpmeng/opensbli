#!/bin/bash
##@brief Download and install a specific version of CMake
##@author Jianping Meng
##@contributors
##@details

function usage {
    echo "This script will download and install CMake to a specified directory!"
    echo "./$(basename $0) -h -> Showing usage"
    echo "./$(basename $0) -d -> Specifying the directory for installation"
    echo "./$(basename $0) -v -> Specifying the version (3.20.0 by default)"
    echo "./$(basename $0) -s -> enable installation in the system directory (e.g., /usr/local/)"
    echo "The default directory for installation is \$HOME/CMake in user directory and /usr/local/CMake in system directory"
    echo "if choosing system directory to install, sudo is needed"
}
optstring="hd:v:s"
User="Y"
Version="3.20.0"

while getopts ${optstring} options; do
    case ${options} in
        h)
            usage
            exit 0
        ;;
        d)
            Dir=${OPTARG}
        ;;
        v)
            Version=${OPTARG}
        ;;
        s)
            User="N"
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

if [ -z "$Dir" ]
then
   if [[ "$User" == "Y" ]]
   then
       Dir="$HOME/CMake"
   fi
   if [[ "$User" == "N" ]]
   then
       Dir="/usr/local/CMake"
   fi
fi

if [ $# -eq 0 ]
then
    echo "This script will download and install CMake-$Version to ${Dir}!"
fi

wget -c https://github.com/Kitware/CMake/releases/download/v$Version/cmake-$Version-Linux-x86_64.sh

if [ ! -d "$Dir" ]
then
    rm -r -f "$Dir"
fi
mkdir $Dir
sh ./cmake-$Version-Linux-x86_64.sh --prefix=$Dir  --skip-license

if [[ "$User" == "Y" ]]
then
    if [ ! -d "$HOME/bin" ]
    then
        mkdir -p $HOME/bin
    fi
    if [  -f "$HOME/bin/cmake" ]
    then
        rm $HOME/bin/cmake
    fi
    ln -s $Dir/bin/cmake $HOME/bin/cmake
    echo "Please add $HOME/bin to path for enabling CMake"
fi

if [[ "$User" == "N" ]]
then
   if [ ! -d "/usr/local/bin" ]
   then
        mkdir -p /usr/local/bin
   fi
   if [  -f "/usr/local/bin/cmake" ]
   then
       rm -r -f /usr/local/bin/cmake
   fi
   ln -s $Dir/bin/cmake /usr/local/bin/cmake
fi

rm cmake-$Version-Linux-x86_64.sh