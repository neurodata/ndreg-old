function anno = getAnno(server, token, resolution, xRange, yRange, zRange)
setup

annoSize = getAnnoSize(server, token);
if ~exist('resolution','var')
   resolution = 0;
end

if ~exist('xRange','var') | isempty(xRange)
   xRange = [0 annoSize(1)]; % Number of columns
end

if ~exist('yRange','var') | isempty(yRange)
   yRange = [0 annoSize(2) ]; % Number of rows
end

if ~exist('zRange','var') | isempty(zRange)
   zRange = [0 annoSize(3)]; % Number of slices
end


oo = OCP();
oo.setServerLocation(server);
oo.setAnnoToken(token);
oo.setDefaultResolution(resolution);

q = OCPQuery;
q.setType(eOCPQueryType.annoDense);
q.setResolution(resolution);
q.setCutoutArgs(xRange, yRange, zRange);
ramonVol = oo.query(q);
anno = ramonVol.data;
