#!/bin/bash
##@brief Create  a working environment for OpenSBLI!
##@author Jianping Meng
##@contributors
##@details

function usage {
    echo "This script will download, compile and install the OPS library to a specified directory!"
    echo "./$(basename $0) -h -> Showing usage"
    echo "./$(basename $0) -d -> Specifying the directory (absolute path) for creating the environment"
    echo "./$(basename $0) -c -> Specifying the compiler"
    echo "./$(basename $0) -m -> Specifying the machine type"
    echo "Machine type can be: Ubuntu (default) ARCHER2 IRIDIS5 Fedora"
}
optstring=":dcmh"
Compiler="Gnu"
Dir="$HOME/OPS_INSTALL"
Machine="Ubuntu"

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

mkdir -p $1
cd $1
wget -c https://github.com/jpmeng/utilities/archive/refs/heads/main.zip
unzip -j main.zip
rm main.zip
./InstallPython2.sh $1/Python2
./InstallOPS.sh $1/OPS-INSTALL $2
wget -c https://github.com/opensbli/opensbli/archive/refs/heads/cpc_release.zip
unzip cpc_release.zip
rm cpc_release.zip