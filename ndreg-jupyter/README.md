This Dockerfile builds a fully-functional jupyter with support for Neurodata services.

To build a new Docker image:
docker build . -t ndreg-jupyter

To run and test with a sample notebook on port 8888:
docker run -p 8888:8888 ndreg-jupyter

If you want to save notebooks, you will need to mount a local path, e.g:
docker run -p 8888:8888  -v `pwd`/notebooks:/run/notebooks ndreg-jupyter
