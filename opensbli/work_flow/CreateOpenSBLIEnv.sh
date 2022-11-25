##@brief Create  a working environment for OpenSBLI!
##@author Jianping Meng
##@contributors
##@details
#!/bin/bash
if [ $# -ne 2 ]
then
    echo "This script will create the whole OpenSBLI environment !"
    echo "Usage: ./CreateOpenSBLIEnv.sh Dir"
    echo "Dir: the directory for the environment. Note: Dir must be a absoulte path"
    echo "Machine type: Ubuntu ARCHER2"
    exit 1
fi
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


