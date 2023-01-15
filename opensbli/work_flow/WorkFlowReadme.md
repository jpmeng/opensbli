# How to use work flow scripts

- Step 1 Create the environment

Copy CreateOpenSBLIEnv.sh into a directory, make it executable and run, for example, to create an environment including Python2, OPS.

We note that the first argument of the script (i.e., the directory) must use absolute path.

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
- Step 3 Compile the C/C++ code

```bash
# Under the environment directory
cd opensbli-cpc_release/apps/transitional_SBLI/
../../../CompileC.sh ~/tmp/OpenSBLIEnv ARCHER2
```

- Tips
  - There is detail explanation on usage in each script