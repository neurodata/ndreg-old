function imageSize = getImageSize(server, token)
setup

ocp = OCP();
q = OCPQuery;
q.setType(eOCPQueryType.imageDense);
ocp.setServerLocation(server);
ocp.setImageToken(token);

resolution = 0;
ranges = ocp.imageInfo.DATASET.IMAGE_SIZE(resolution); 

nRows = ranges(1);
nCols = ranges(2);
nSlices = ocp.imageInfo.DATASET.SLICERANGE(2) - ocp.imageInfo.DATASET.SLICERANGE(1);
imageSize = [nRows, nCols, nSlices];


