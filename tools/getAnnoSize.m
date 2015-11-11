function annoSize = getAnnoSize(server, token, resolution)
if ~exist('resolution','var')
   resolution = 0;
end
setup

ocp = OCP();
q = OCPQuery;
q.setType(eOCPQueryType.annoDense);
ocp.setServerLocation(server);
ocp.setAnnoToken(token);

ranges = ocp.annoInfo.DATASET.IMAGE_SIZE(resolution); 

nRows = ranges(1);
nCols = ranges(2);
nSlices = ocp.annoInfo.DATASET.SLICERANGE(2) - ocp.annoInfo.DATASET.SLICERANGE(1);
annoSize = [nRows, nCols, nSlices];


