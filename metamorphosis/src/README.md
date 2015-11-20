# ndreg metamorphosis
Executable for doing simultaneous diffeomorphic registration (LDDMM) and intensity correction.
itkCommandLineArgumentParser.h* come from https://github.com/midas-journal/midas-journal-793.
Submit bug reports [here](https://github.com/openconnectome/ndreg/issues/new).

## Build Requirements
- ITK 4.0
- CMake 2.6 
- FFTW (recomended)

## Building (On Ubuntu)
```bash
sudo apt-get install libinsighttoolkit4-dev fftw3 cmake-curses-gui # Install required packages
mkdir ../bin
cd ../bin
ccmake -G "Unix Makefiles" ../src/ # Configure and generate build using console
make
```

## Changelog
- **1.0** (November 20, 2015)
    - Initial commit of metamorphosis code
