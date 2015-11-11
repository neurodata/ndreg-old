function putImage(image, token, channel, resolution, server)
addpath('../Utilities/NIFTI_20110921/')
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
oo.setImageToken(token);
oo.setDefaultResolution(resolution);

if exist('channel','var')
   oo.setImageChannel(channel);
end

% Resize image so that it fits dataset on OCP server
[inNumRows, inNumCols, inNumSlices] = size(image);

outSize = oo.imageInfo.DATASET.IMAGE_SIZE(resolution);
outNumRows = outSize(2);
outNumCols = outSize(1);
outNumSlices = outSize(3);

numRows = min(inNumRows,outNumRows);
numCols = min(inNumCols,outNumCols);
numSlices = min(inNumSlices, outNumSlices);
image = image(1:numRows,1:numCols, 1:numSlices);

% Upload image in blocks
rowStep = 128;
colStep = 128;
sliceStep = 128;

for slice = 1:sliceStep:numSlices
    for row = 1:rowStep:numRows
        for col = 1:colStep:numCols
            disp(sprintf('[%d,%d,%d]',row,col,slice)) 

            if (row+rowStep) < numRows
                i = row;
            else
                i = numRows-rowStep;
            end

            if (col+colStep) < numCols
                j = col;
            else
                j = numCols-colStep;
            end

            if (slice+sliceStep) < numSlices; 
                k = slice;
            else
                k = numSlices - sliceStep;
            end

            xyzOffset = [j-1, i-1, k-1];
            block = image(i:(i+rowStep), j:(j+colStep), k:(k+sliceStep));
            paint = RAMONVolume();
            paint.setResolution(resolution);
            paint.setXyzOffset(xyzOffset);
            paint.setCutout(block);
            paint.setChannel(channel);

            try
                oo.uploadImageData(paint);
            catch ME
                disp('Failed to upload block.')
                return;
            end

        end
    end
end
