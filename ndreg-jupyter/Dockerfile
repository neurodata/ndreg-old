FROM ubuntu:16.04
MAINTAINER Eric Perlman / Neurodata (neurodata.io)

RUN apt-get update
RUN apt-get -y upgrade 

RUN apt-get -y install build-essential

RUN apt-get -y install \
  python-pip \
  python-all-dev \
  zlib1g-dev \
  libjpeg8-dev \
  libtiff5-dev \
  libfreetype6-dev \
  liblcms2-dev \
  libwebp-dev \
  tcl8.5-dev \
  tk8.5-dev \
  python-tk \
  libhdf5-dev \
  git \
  cmake \
  libinsighttoolkit4-dev \
  libfftw3-dev

RUN pip install --upgrade pip
RUN pip install matplotlib SimpleITK
RUN pip install numpy
RUN pip install ndio

# We currently following 'master' to incorperate many recent bug fixes.
# When stable, use the following instead:
# RUN pip install intern
WORKDIR /work
RUN git clone https://github.com/jhuapl-boss/intern.git /work/intern --single-branch
WORKDIR /work/intern
RUN python setup.py install

# Set up ipython
RUN pip install ipython
RUN pip install ipython[all]
RUN pip install jupyter

# Build ndreg. Cache based on last commit.
WORKDIR /work
ADD https://api.github.com/repos/neurodata/ndreg/git/refs/heads/master version.json
RUN git clone https://github.com/neurodata/ndreg.git /work/ndreg --branch master --single-branch
WORKDIR /work/ndreg
RUN cmake . && make && make install

# Add Tini
ENV TINI_VERSION v0.15.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

WORKDIR /run

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
