FROM neurodata/ndio
MAINTAINER Kwame Kutten / Neurodata (neurodata.io)

USER root

RUN apt-get clean 
RUN apt-get update
RUN apt-get -y upgrade 

RUN apt-get -y install cmake \
	libinsighttoolkit4-dev \
	libfftw3-dev

RUN pip install simpleitk matplotlib

WORKDIR /home/nd-user
RUN git clone https://github.com/neurodata/ndreg.git
WORKDIR /home/nd-user/ndreg
RUN cmake . && make && make install

USER nd-user

ENTRYPOINT ["/bin/bash"]
