##@brief Translate a OpenSBLI app to OPS C/C++ code
##@author Jianping Meng
##@contributors
##@details
#!/bin/bash
if [ $# -ne 2 ]
then
    echo "This script will create the whole OpenSBLI environment !"
    echo "Usage: ./translate.sh EnvDir source "
    echo "EnvDir: the directory for the environment."
    echo "src: python source file"
    exit 1
fi
EnvDir=$1
src=$2
 # activate Python2
source $1/Python2/bin/activate
export PYTHONPATH=$PYTHONPATH:$1/opensbli-cpc_release
python $src