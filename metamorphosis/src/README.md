# ndreg metamorphosis
Executable for doing simultaneous diffeomorphic registration (LDDMM) and intensity correction.
itkCommandLineArgumentParser.h* come from https://github.com/midas-journal/midas-journal-793.
Submit bug reports [here](https://github.com/openconnectome/ndreg/issues/new).

## Requirements
- ITK 4.0
- CMake 2.6 
- FFTW

## Building (On Ubuntu 14.04)
```bash
sudo apt-get install libinsighttoolkit4-dev libfftw3-dev libtiff5-dev # Install required packages
sudo apt-get install cmake-curses-gui # Install recomended packages
mkdir ../bin
cd ../bin
cmake -G "Unix Makefiles" ../src/ # Configure and generate build using console
make
```

## Changelog
- **1.0** (November 20, 2015)
    - Initial commit of metamorphosis code
