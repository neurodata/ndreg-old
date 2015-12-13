# ndreg
NeuroData's registration library, built in python using ITK

Submit bug reports [here](https://github.com/openconnectome/ndreg/issues/new).


## Requirements
- ndio (https://github.com/openconnectome/ndio.git)
- ITK 4.0
- SimpleITK
- numpy 
- FFTW (recomended)

## Installation (On Ubuntu 14.04)

1. Install numpy and SimpleITK
```bash
sudo apt-get install python-numpy
easy_install --user SimpleITK
```
1. Build metamorphosis binary for diffeomorphic registration using instructions in ./metamorphosis/src/README.md

## Changelog
- **0.3** (December 4, 2015)
    - Now using ndio to download images from OCP (Issue #4)
    - Added jupyter notebook demonstrating how to run mask pipeline (Issue #1)

- **0.2** (November 20, 2015)
    - Replaced ITK with SimpleITK in python code
    - Added diffeomorphic registration

- **0.1** (November 11, 2015)
    - Initial commit for Rigid/Affine registration

## Under Development
- [ ] Add support for ndio
