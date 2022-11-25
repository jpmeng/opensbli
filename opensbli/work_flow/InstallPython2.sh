##@brief Create a Python2 environment
##@author Jianping Meng
##@contributors
##@details
#!/bin/bash
if [ $# -eq 0 ]
then
    dir="$HOME/Python2"
    echo "This script will download and install the Python2 (miniconda2) to a specified directory!"
    echo "Please enter the directory for it..."
    echo "if without specifying the directory, we will use $dir"
else
    dir=$1
fi
wget -c https://repo.anaconda.com/miniconda/Miniconda2-py27_4.8.3-Linux-x86_64.sh
chmod a+x ./Miniconda2-py27_4.8.3-Linux-x86_64.sh
./Miniconda2-py27_4.8.3-Linux-x86_64.sh -p $dir -b
source $dir/bin/activate
pip install sympy==1.1
pip install numpy matplotlib h5py scipy
 if [ ! -d "$HOME/bin" ]
    then
        mkdir -p $HOME/bin
fi
rm Miniconda2-py27_4.8.3-Linux-x86_64.sh
cd $HOME/bin
echo "#!/bin/bash" > Python2
echo "echo \"The default shell for Python2 is bash. Enter the shell name to change\" " >> Python2
echo "if [ \$# -eq 0 ]" >> Python2
echo "then" >> Python2
echo "    shell=\"bash\" " >> Python2
echo "else" >> Python2
echo "    shell=\$1" >> Python2
echo "fi" >> Python2
echo "source $dir/bin/activate" >> Python2
echo "\$shell" >> Python2
chmod a+x Python2
echo "$HOME/bin/Python2 can be called to activate the Python2 environment!"
echo "Consider to add $HOME/bin to path"



