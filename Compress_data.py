import os
import zipfile
import numpy as np
import nibabel as nib
import gzip


def dfs_get_zip_file(input_path,result):
    files = os.listdir(input_path)
    for file in files:
        if os.path.isdir(input_path+'/'+file):
            dfs_get_zip_file(input_path+'/'+file, result)
        else:
            result.append(input_path+'/'+file)

def zip_path(input_path, output_path, output_name):
    f = zipfile.ZipFile(output_path+'/'+output_name,'w',zipfile.ZIP_DEFLATED)
    filelists = []
    dfs_get_zip_file(input_path, filelists)
    for file in filelists:
        f.write(file)
    f.close()


def zip_file(tar_file, zipped_file):
    f = zipfile.ZipFile(zipped_file, 'w', zipfile.ZIP_DEFLATED)
    f.write(tar_file)
    f.close()

def extract_zip_file(zipped_file):
    zfile = zipfile.ZipFile(zipped_file, 'r')
    for filename in zfile.namelist():
        zfile.extract(filename, path= '/')
    zfile.close()

def extract_zip_file_to_current_path(zipped_file):
    unZf = zipfile.ZipFile(zipped_file, 'r')
    targetDir, _ = os.path.split(zipped_file)
    for filename in unZf.namelist():
        originalDir, name = os.path.split(filename)
        unZfTarge = os.path.join(targetDir, name)
        if unZfTarge.endswith("/"):
            #empty dir
            splitDir = unZfTarge[:-1]
            if not os.path.exists(splitDir):
                os.makedirs(splitDir)
        else:
            splitDir, _ = os.path.split(targetDir)
            if not os.path.exists(splitDir):
                os.makedirs(splitDir)
            hFile = open(unZfTarge,'wb')
            hFile.write(unZf.read(filename))
            hFile.close()
    unZf.close()

def read_zipped_volume(zipped_file, Volume_size, data_type):
    # extract_zip_file(zipped_file)
    extract_zip_file_to_current_path(zipped_file)
    fid = open(zipped_file[:-4], 'rb')
    image = np.fromfile(fid, dtype=data_type, count=int(Volume_size[0]) * int(Volume_size[1]) * int(Volume_size[2]))
    fid.close()
    image = image.reshape([Volume_size[0], Volume_size[1], Volume_size[2]])
    os.remove(zipped_file[:-4])
    return image

def save_zipped_volume(Volume, zipped_file):
    save_file = zipped_file[:-4]
    Volume.tofile(save_file)
    zip_file(save_file, zipped_file)
    os.remove(save_file)


def gzip_nii(fpath):
    f_in = open(fpath, 'rb')
    f_out = gzip.open(fpath+'.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()

def save_nii(image, affine, fpath, keep_nii=False):
    image_nii = nib.Nifti1Image(image, affine)
    image_header = image_nii.header
    image_header['regular'] = 'r'
    image_header['glmax'] = np.double(np.max(image))
    image_header['glmin'] = np.double(np.min(image))
    image_header['sform_code'] = 0
    nib.save(image_nii, fpath)
    gzip_nii(fpath)
    if keep_nii is False and os.path.exists(fpath):
        os.remove(fpath)

def read_nii(fpath, keep_nii= False):
    # if '.gz' in fpath:
    #     f_out = open(fpath[:-3], 'wb')
    #     f_in = gzip.open(fpath, 'rb')
    #     f_out.writelines(f_in)
    #     f_out.close()
    #     f_in.close()
    #     nii_data = nib.load(fpath[:-3])
    # else:
    #     nii_data = nib.load(fpath)
    nii_data = nib.load(fpath)
    image = nii_data.get_fdata()
    affine = nii_data.affine
    image = np.squeeze(image)
    return image, affine