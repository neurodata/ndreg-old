function image = getImage(server, token, resolution, xRange, yRange, zRange)
setup

imageSize = getImageSize(server, token);
if ~exist('resolution','var')
   resolution = 0;
end

if isempty(xRange) | ~exist('xRange','var') 
   xRange = [0 imageSize(1)]; % Number of columns
end

if isempty(yRange) | ~exist('yRange','var') 
   yRange = [0 imageSize(2)]; % Number of rows
end

if isempty(zRange) | ~exist('zRange','var') 
   zRange = [0 imageSize(3)]; % Number of slices
end


oo = OCP();
oo.setServerLocation(server);
oo.setImageToken(token);
oo.setDefaultResolution(resolution);

q = OCPQuery;
q.setType(eOCPQueryType.imageDense);
q.setResolution(resolution);
q.setCutoutArgs(xRange, yRange, zRange);
q.setChannels({'Grayscale'});
ramonVol = oo.query(q);
image = ramonVol{1}.data;


