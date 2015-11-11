function imgWrite(img, path, spacing)
  addpath('/cis/home/kwame/matlab/NIFTI_20110921/')
  if nargin < 3
     spacing = [1,1,1];
  end

  [~,~,ext] = fileparts(path);
  if strcmp(ext,'.img') | strcmp(ext,'.hdr')
    ana = make_ana(img);
    ana.hdr.dime.pixdim(2:4) = spacing;
    save_untouch_nii(ana, path);
  else
    fid = fopen(path,'w');
    fid = fwrite(fid, img);
  end
end
