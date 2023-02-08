#!/bin/bash
##@brief Create a Python environment
##@author Jianping Meng
##@contributors
##@details

function usage {
    echo "This script will download, compile and install the OPS library to a specified directory!"
    echo "./$(basename $0) -h -> Showing usage"
    echo "./$(basename $0) -d -> Specifying the directory (absolute path) for installing Python"
    echo "./$(basename $0) -p -> Use Python2"
}
optstring="hd:p"
Dir="$HOME/Python"
PackageName="Miniconda3-py37_22.11.1-1-Linux-x86_64.sh" # 3.7

while getopts ${optstring} options; do
    case ${options} in
        h)
            usage
            exit 0
        ;;
        d)
            Dir=${OPTARG}
        ;;
        p)
            PackageName="Miniconda2-py27_4.8.3-Linux-x86_64.sh"
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

if [ -d "$Dir" ]
then
   echo "The specified directory is exising, please delete it first or change to another one!"
   exit 1
fi

wget -c https://repo.anaconda.com/miniconda/${PackageName}
chmod a+x ./${PackageName}
./${PackageName} -b -p ${Dir}
source "$Dir/bin/activate" "$Dir"
pip install sympy==1.1
pip install numpy matplotlib h5py scipy Jinja2

#  if [ ! -d "$HOME/bin" ]
#     then
#         mkdir -p $HOME/bin
# fi
rm ${PackageName}
pip cache purge
#cd $HOME/bin
# echo "#!/bin/bash" > Python2
# echo "echo \"The default shell for Python2 is bash. Enter the shell name to change\" " >> Python2
# echo "if [ \$# -eq 0 ]" >> Python2
# echo "then" >> Python2
# echo "    shell=\"bash\" " >> Python2
# echo "else" >> Python2
# echo "    shell=\$1" >> Python2
# echo "fi" >> Python2
# echo "source $dir/bin/activate" >> Python2
# echo "\$shell" >> Python2
# chmod a+x Python2
# echo "$HOME/bin/Python2 can be called to activate the Python2 environment!"
# echo "Consider to add $HOME/bin to path"