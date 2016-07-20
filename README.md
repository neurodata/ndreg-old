# ndreg
NeuroData's registration library, built in python using ITK

Submit bug reports [here](https://github.com/openconnectome/ndreg/issues/new).


## Installating dependances on Ubuntu 16.04 and later
 * Install dependancies

```bash
sudo apt-get -y install cmake python-numpy libinsighttoolkit4-dev libfftw3-dev
```
 * Proceded to **Installing ndreg**

## Installing dependances other linux distributions
 * Build and install ITK 4.10.0 or later
 
 ```bash
 itkVersion=4.10.0
 itkMinorVersion=`echo ${itkVersion} | cut -d'.' -f 1,2`
 mkdir itk; cd itk
 wget https://sourceforge.net/projects/itk/files/itk/${itkMinorVersion}/InsightToolkit-${itkVersion}.tar.gz
 tar -vxzf InsightToolkit-${itkVersion}.tar.gz
 mv InsightToolkit-${itkVersion} src/
 mkdir bin; cd bin
 cmake -G "Unix Makefiles" -DITK_USE_SYSTEM_FFTW=OFF -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DModule_ITKReview=ON ../src 
 make && sudo make install
 ```
 
 * For Ubuntu 15.10 and earlier
 
  * Install dependances

  ```bash
  sudo apt-get -y install cmake
  ```
  * Proceded to **Installing ndreg**

 * For Centos 6.7 and later
  * Install dependances
  ```bash
  sudo yum -y install libjpeg-devel cmake 
  sudo /usr/local/bin/pip install  --trusted-host www.simpleitk.org -f http://www.simpleitk.org/SimpleITK/resources/software.html SimpleITK
  sudo /usr/local/bin/pip install blosc==1.3.0
  ```
  * Proceded to **Installing ndreg**

## Installing ndreg
```
git clone https://github.com/neurodata/ndreg.git
cd ndreg
cmake .
make && sudo make install
```
