# How to use work flow scripts

- Step 1 Create the environment

Copy CreateOpenSBLIEnv.sh into a directory, make it executable and run, for example, to create an environment including Python2, OPS.

We note that the first argument of the script (i.e., the directory) must use absolute path.

Currently only the OPS library and the OpenSBLI framework are installed inside the directory.

```bash
./CreateOpenSBLIEnv.sh ~/tmp/OpenSBLIEnv
```

- Step 2 Translate an application to C/C++

```bash
# Under the environment directory
cd opensbli-cpc_release/apps/transitional_SBLI/
../../../Translate.sh ~/tmp/OpenSBLIEnv transitional_SBLI.py
../../../CompileC.sh ~/tmp/OpenSBLIEnv ARCHER2
```

Step 3 Compile the C/C++ code

```bash
# Under the environment directory
cd opensbli-cpc_release/apps/transitional_SBLI/
../../../CompileC.sh ~/tmp/OpenSBLIEnv ARCHER2
```

- Other utilities
- - CMake
    InstallCMake.sh can help to download and install CMake with a specified version. Assuming we would like to install Version 3.22.2

    ```bash
    # install CMake into /usr/local/CMake
    sudo ./InstallCMake.sh -v 3.22.2 -s
    # install CMake into $HOME/CMake
    ./InstallCMake.sh -v 3.22.2
    # install CMake into $HOME/tmp/CMake
    ./InstallCMake.sh -v 3.22.2 -d $HOME/tmp/CMake

    ```
    After installation, the script will add the a link to cmake into /usr/loca/bin/ or $HOME/bin
- Tips

  - There is detail explanation on usage in each script, which can be shown by, e.g.,

    ```bash
      ./InstallCMake.sh -h
    ```
