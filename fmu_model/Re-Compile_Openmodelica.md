
# Compile on MacOS

- open a terminal window and navigate to the directory containing the original FMU file

- greate a directory to extract the contents of the FMU file:

```bash
mkdir <fmu-file>
cd <fmu-file>
```

- run the following command to extract the contents of the FMU file:

```bash
unzip -o ../<fmu-file>.fmu
```

- navigate to the extracted directory

- copy the interface files directory to the extracted directory (they are available online at [FMI standard](https://fmi-standard.org/))

- make the necessary changes to the source code

### List of changes:

-  in **`CMakelists.txt`**, set the GNU compiler:

    ```cmake
    # use gcc and g++ compiler
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
    set(CMAKE_C_COMPILER_ID "GNU")
    ```

- set the optimization level in **`CMakelists.txt`**:

    ```cmake
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
    ```

- set the path to the FMI header files in **`CMakelists.txt`**:

    ```cmake
    set(FMI_INTERFACE_HEADER_FILES_DIRECTORY

    ${CMAKE_CURRENT_SOURCE_DIR}/../fmi2

    CACHE STRING

    "Path to FMI header files containing fmi2Functions.h, fmi2FunctionTypes.h, fmi2TypesPlatforms.h")
    ```
- in **`CMakelists.txt`**, comment out the following line

    ```cmake
    # target_link_options(${FMU_NAME} PRIVATE "LINKER:SHELL:-undefined,error")
    ```

- in **`CMakelists.txt`**, add the following line to the ```install(TARGETS ...```

    ```cmake
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ```

>[!WARNING]
In the case the FMU is generated with the Directional Derivatives implementation, the following changes have to be made to the source code:

- find and fix the datatype conflicts of the ```data``` argument of the following functions from
    ```c
    int xy_model_initialAnalyticJacobianFMIDERINIT(DATA* data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian);
    int xy_model_initialAnalyticJacobianFMIDER(DATA* data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian);
    ```
    to
    ```c
    int xy_model_initialAnalyticJacobianFMIDERINIT(void* data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian);
    int xy_model_initialAnalyticJacobianFMIDER(void* data, threadData_t *threadData, ANALYTIC_JACOBIAN *jacobian);
    ```
    both in the function signatures and definitions.

- create a build directory:

```bash
mkdir build
cd build
```

- run the following command to compile the source code:

```bash
cmake ..
make -j
```

- copy the shared library to the binary directory:

```bash
mkdir -p ../../binaries/darwin64
cp <FMU_NAME>.dylib ../../binaries/darwin64
```

## Repackage the FMU

- navigate to the root directory of the FMU

```bash
cd ../../
```

- run the following command to create the FMU file:

```bash
zip -r -0 ../<FMU_NAME>.fmu *
```

- the FMU file is now ready to be used in a simulation environment