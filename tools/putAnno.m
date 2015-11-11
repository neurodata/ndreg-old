function putAnno(annoImage, annoToken, annoChannel, resolution, server)
setup

if ~exist('server','var') 
    server = 'openconnecto.me';
end

if ~exist('resolution','var')
   resolution = 0;
end

% Setup OCP interface
oo = OCP();
oo.setServerLocation(server);
oo.setAnnoToken(annoToken);
oo.setDefaultResolution(resolution);

if exist('annoChannel','var')
    oo.setAnnoChannel(annoChannel);
end


% Resize annoImage so that it fits dataset on OCP server
[inNumRows, inNumCols, inNumSlices] = size(annoImage);

outSize = oo.annoInfo.DATASET.IMAGE_SIZE(resolution);
outNumRows = outSize(2);
outNumCols = outSize(1);
outNumSlices = outSize(3);

numRows = min(inNumRows,outNumRows);
numCols = min(inNumCols,outNumCols);
numSlices = min(inNumSlices, outNumSlices);
annoImage = annoImage(1:numRows,1:numCols, 1:numSlices);

% Upload annotation volumes
disp('Uploading annotation volumes...');
uid = unique(annoImage); % Create lists of unique IDs
for i = 1:length(uid)
   if uid(i) == 0
       continue
   end
   %disp(uid(i))
   seg = RAMONSegment();
   seg.setId(uid(i));
   seg.setResolution(resolution);
   oo.createAnnotation(seg);
end

% Upload annotation paint volume
disp('Uploading paint volume...')
paint = RAMONVolume();
paint.setCutout(annoImage);
paint.setResolution(resolution);
paint.setChannel(annoChannel);
paint.setXyzOffset([0,0,0]);
paint.setDataType(eRAMONChannelDataType.uint32);
paint.setChannelType(eRAMONChannelType.annotation);

try 
    oo.createAnnotation(paint);
catch ME
      disp('Failed to upload annotation.')
      return;
end
