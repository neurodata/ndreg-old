#!/bin/bash

cd ~
# Install ndio
sudo /usr/local/bin/pip install blosc==1.3.0
sudo yum -y install libjpeg-devel
sudo /usr/local/bin/pip install ndio

# Build and install ITK
itkVersion=4.10.0
itkMinorVersion=`echo ${itkVersion} | cut -d'.' -f 1,2`
mkdir itk; cd itk
wget https://sourceforge.net/projects/itk/files/itk/${itkMinorVersion}/InsightToolkit-${itkVersion}.tar.gz
tar -vxzf InsightToolkit-${itkVersion}.tar.gz
mv InsightToolkit-${itkVersion} src/
mkdir bin; cd bin
cmake -G "Unix Makefiles" -DITK_USE_SYSTEM_FFTW=OFF -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DModule_ITKReview=ON ../src 
make && sudo make install
cd ../.. && rm -rf itk #Clean up

# Install SimpleITK
sudo /usr/local/bin/pip install  --trusted-host www.simpleitk.org -f http://www.simpleitk.org/SimpleITK/resources/software.html SimpleITK

# Build and install ndreg
mkdir ndreg; cd ndreg
git clone https://github.com/neurodata/ndreg.git ./src
mkdir bin; cd bin
cmake -G "Unix Makefiles" ../src
make && make install

