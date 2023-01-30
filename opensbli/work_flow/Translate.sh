#!/bin/bash
##@brief Translate a OpenSBLI app to OPS C/C++ code
##@author Jianping Meng
##@contributors
##@details

function usage {
    echo "This script will translate the OpenSBLI Python cod  e to C/C++!"
    echo "Usage: ./$(basename $0) PythonSource"
    echo "./$(basename $0) -h -> showing usage"  
    echo "./$(basename $0) -e -> Specifying the environment directory."
}

ScriptPath="${BASH_SOURCE:-$0}"
AbsolutScriptPath="$(realpath "${ScriptPath}")"
EnvDir="$(dirname "${AbsolutScriptPath}")"

optstring="e:h"


while getopts ${optstring} options; do
    case ${options} in
        h)
            usage
            exit 0
        ;;       
        e)
            EnvDir=${OPTARG}
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

Src=${@:$OPTIND:1}

if [ -z $Src ]
then
    echo "This script will translate the OpenSBLI Python code to C/C++!"
    echo "Usage: ./$(basename $0) [options] PythonSource"
    echo "./$(basename $0) -h -> showing usage"
    echo "./$(basename $0) -e -> Specifying the environment directory."
fi


echo "Source name is ${Src} and evn is ${EnvDir}"
# activate Python
source ${EnvDir}/Python/bin/activate "${EnvDir}/Python"
export PYTHONPATH=$PYTHONPATH:$EnvDir/OpenSBLI
python $Src