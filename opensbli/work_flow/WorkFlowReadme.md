# How to use work flow scripts

> These set scripts are intended to automatically set up an environment both using and developing OpenSBLI. However, they could be also utilised separately for installing [OPS](https://github.com/OP-DSL/OPS) library, HDF5 library, and create an independent Python environment. 

- Tools needed for scripts
  These work flow scripts will require the following tools pre-installed:

  - unzip
  - wget
  - sed
  - CMake

    Since OPS will need CMake version higher than 3.18, the default CMake provided by the system might not be good. Therefore we provide InstallCMake.sh to download and install CMake with a specified version. Assuming we would like to install Version 3.22.2

    ```bash
    # install CMake into /usr/local/CMake
    sudo ./InstallCMake.sh -v 3.22.2 -s
    # install CMake into $HOME/CMake
    ./InstallCMake.sh -v 3.22.2
    # install CMake into $HOME/tmp/CMake
    ./InstallCMake.sh -v 3.22.2 -d $HOME/tmp/CMake

    ```
    After installation, the script will add the a link to cmake into /usr/loca/bin/ or $HOME/bin
- Step 1 Create the environment

  Copy CreateOpenSBLIEnv.sh into a directory, make it executable and run, for example, to create an environment including Python2, OPS.

  We note that the first argument of the script (i.e., the directory) must use absolute path.

  By default, the OPS library, the Python environment, the OpenSBLI framework are installed inside the directory. If required, a local HDF5 can also be installed.

  ```bash
  ./CreateOpenSBLIEnv.sh -d ~/tmp/OpenSBLIEnv
  ```
- Step 2 Translate an application to C/C++

  By default, Translate.sh will set the directory where it is to be the environment directory. It will activate the Pyhon environment in the environment, and tranlsate the specified OpenSBLI Python source code.

  ```bash
  # Under the environment directory
  cd opensbli-cpc_release/apps/transitional_SBLI/
  ../../../Translate.sh transitional_SBLI.py
  ../../../CompileC.sh ~/tmp/OpenSBLIEnv ARCHER2
  ```
- Step 3 Compile the C/C++ code

  When creating the environment at Step 1, the default machine in CompileC.sh will be set to the intended one. Also, CompileC.sh will set the directory where it is to be the environment directory, and use the local HDF5 at the first instance.

  ```bash
  # Under the environment directory
  cd opensbli-cpc_release/apps/transitional_SBLI/
  ../../../CompileC.sh
  ```
- Tips

  - There is detail explanation on usage in each script, which can be shown by, e.g.,

    ```bash
      ./InstallCMake.sh -h
    ```
