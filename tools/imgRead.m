function [img, spacing] = imgRead(path, precision)
  addpath('../Utilities/NIFTI_20110921/');
  if nargin < 2
     precision = 'uint8';
  end

  [~,~,ext] = fileparts(path);
  if strcmp(ext,'.img') | strcmp(ext,'.hdr')
    nii = load_untouch_nii(path);
    img = nii.img;
    spacing = nii.hdr.dime.pixdim(2:4);
  else
    spacing = [1,1,1];
    fid = fopen(path,'r');
    img = fread(fid,Inf,precision);
    fclose(fid);
  end
end
