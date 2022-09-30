from torch.utils.data import Dataset
import torch
import glob
import math
import os
import SimpleITK as sitk
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import multiprocessing
from joblib import Parallel, delayed
import sys
import json
from Compress_data import read_nii, save_nii
# multiprocessing.set_start_method('forkserver', force=True)
import gc
import numpy as np
from ImageProcessingSitk import Dcm_improcessing_sitk

# pynvml.nvmlInit()

sys.setrecursionlimit(100000)


def numpy_to_tensor(img):
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    return img


def print3Dnumpy(savefolder, flag, npy3D):
    import cv2
    import os
    if len(npy3D.shape) > 2:
        savepath = os.path.join(savefolder, flag)
        if len(npy3D.shape) == 4:
            temp = npy3D[0, :, :, :]
        else:
            temp = npy3D

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        for i in range(temp.shape[0]):
            img = temp[i, :, :].squeeze()
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())

            img = 255 * img
            img = np.asarray(img, np.uint8)
            cv2.imwrite(os.path.join(savepath, "{}.png".format(i)), img)
    else:
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        img = npy3D
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        elif img.max() > 0:
            img = img / img.max()

        img = 255 * img
        img = np.asarray(img, np.uint8)
        cv2.imwrite(os.path.join(savefolder, "{}.png".format(flag)), img)


def write_dict_json(jsonname, dictObj):
    jsObj = json.dumps(dictObj, indent=4)  # indent参数是换行和缩进
    fileObject = open(jsonname, 'w')
    fileObject.write(jsObj)
    fileObject.close()


def write_list_to_txt(txtpath, inputlist):
    with open(txtpath, "wt") as fp:
        for input in inputlist:
            fp.write(str(input) + "\n")


def read_list_from_txt(txtpath):
    results = []
    with open(txtpath, "rt") as fp:
        lines = fp.readlines()
        for line in lines:
            results.append(line.split("\n")[0])
    return results


def Noise_numpy(src, mask=None, noise_var=0.01, seed=0):
    img = src.copy()
    np.random.seed(seed)
    NOISE_FACTOR = np.random.uniform(0, noise_var) * np.median(img)
    noise = np.random.uniform(0, 1, img.shape)
    if mask is not None:
        img = img + noise * mask * NOISE_FACTOR
    else:
        img = img + noise * NOISE_FACTOR

    img = img / (1.0 + NOISE_FACTOR)

    return img


def crop_resize_slice_based_roi(image_train, mask_train, targets_train, src_thickness, Input_Size, dst_thickness_range, distinguish_left_right=False, margin=4, gpu_id=0):
    D, H, W = image_train.shape

    # 找出所有roi区域
    idx_Z1, idx_Y1, idx_X1 = np.where(mask_train > 0)
    if len(idx_Z1) > 0:
        x0 = random.randint(0, min(idx_X1))
        y0 = random.randint(0, min(idx_Y1))
        x1 = random.randint(min(max(idx_X1), W-1), W - 1)
        y1 = random.randint(min(max(idx_Y1), H-1), H - 1)
    else:
        x0, y0 = 0, 0
        x1, y1 = W - 1, H - 1

    # # roi
    roix = max(random.randint(-50, 0) + x0, 1)
    roiy = max(random.randint(-50, 0) + y0, 1)
    roiw = min(random.randint(0, 50) + x1, W - 1) - roix
    roih = min(random.randint(0, 50) + y1, H - 1) - roiy

    s = min(float(Input_Size[1]) / (y1 - y0), float(Input_Size[2]) / (x1 - x0))
    scale = random.uniform(0.85, 1.301) * s
    # rotate
    angle = random.choice([random.uniform(-20, 20), random.uniform(160, 200)])

    # nslice = 200
    # if D > nslice:
    #     sitker = Dcm_improcessing_sitk()
    #     image_train_temp = []
    #     mask_train_temp = []
    #     targets_train_temp = []
    #     left_up = np.array([roix, roiy, 0],dtype=np.uint32).tolist()
    #     right_down = np.array([roix + roiw, roiy + roih, D], dtype=np.uint32).tolist()
    #     for iz in range(0, D, nslice):
    #         st = iz
    #         ed = min(iz+nslice, D)
    #         Dlen = int(ed - st)
    #         image_train_temp.append(sitker.array_crop_resize_on_roisize_dstsize(image_train[st:ed], left_up,
    #                                                                             right_down,
    #                                                                   (Input_Size[2], Input_Size[1], Dlen),
    #                                                                   interp=sitk.sitkLinear))
    #         targets_train_temp.append(sitker.array_crop_resize_on_roisize_dstsize(targets_train[st:ed], left_up,
    #                                                                               right_down,
    #                                                                     (Input_Size[2], Input_Size[1], Dlen),
    #                                                                     interp=sitk.sitkNearestNeighbor))
    #         mask_train_temp.append(sitker.array_crop_resize_on_roisize_dstsize(mask_train[st:ed], left_up,
    #                                                                  right_down,
    #                                                                  (Input_Size[2], Input_Size[1], Dlen),
    #                                                                  interp=sitk.sitkNearestNeighbor))
    #         image_train = np.concatenate(image_train_temp, axis=0)
    #         mask_train = np.concatenate(mask_train_temp, axis=0)
    #         targets_train = np.concatenate(targets_train_temp, axis=0)
    # else:
    sitker = Dcm_improcessing_sitk()
    left_up = np.array([roix, roiy, 0],dtype=np.uint32).tolist()
    right_down = np.array([roix + roiw, roiy + roih, D], dtype=np.uint32).tolist()
    image_train = sitker.array_crop_resize_on_roisize_dstsize(image_train, left_up, right_down, (Input_Size[2], Input_Size[1], D), interp=sitk.sitkLinear)
    targets_train = sitker.array_crop_resize_on_roisize_dstsize(targets_train, left_up, right_down, (Input_Size[2], Input_Size[1], D), interp=sitk.sitkNearestNeighbor)
    mask_train = sitker.array_crop_resize_on_roisize_dstsize(mask_train, left_up, right_down, (Input_Size[2], Input_Size[1], D), interp=sitk.sitkNearestNeighbor)

    dst_thickness = random.uniform(src_thickness,  max(dst_thickness_range[1], src_thickness) + 0.1)
    thickness_scale = dst_thickness / src_thickness
    image_train, _ = Dcm_improcessing_sitk.array_resize(image_train, thickness_scale, intermethod=sitk.sitkLinear)
    targets_train, _ = Dcm_improcessing_sitk.array_resize(targets_train, thickness_scale, intermethod=sitk.sitkNearestNeighbor)
    mask_train, _ = Dcm_improcessing_sitk.array_resize(mask_train, thickness_scale, intermethod=sitk.sitkNearestNeighbor)
    print('dst_thickness:{}'.format(dst_thickness))

    D, H, W = image_train.shape
    idx_Z, idx_Y, idx_X = np.where( (targets_train > 0) * ( targets_train < 255) )
    if len(idx_Z) > 0:
        z0, z1 = min(idx_Z), max(idx_Z)
    else:
        z0, z1 = 0, D-1

    if margin is not None:
        image_train = image_train[max(z0-margin, 0):min(z1+margin, D)]
        mask_train = mask_train[max(z0-margin, 0):min(z1+margin, D)]
        targets_train = targets_train[max(z0-margin, 0):min(z1+margin, D)]

    return image_train, mask_train, targets_train, dst_thickness



def get_oar_num(oar_list, exclude=[], combine_left_right=False):
    if not combine_left_right:
        return len(oar_list)
    else:
        n = 0
        for oarname in oar_list:
            if oarname not in exclude:
                if '_L' in oarname and oarname.replace('_L', "_R") in oar_list:
                    n += 1

        return len(oar_list) - n

class Normal_Dataset(Dataset):
    def __init__(self, fold_names: list, neg_sample=[], filenamedict=None, exclude_oar=[], batch_size=1, workers=1, augmentation=None,
                 Input_Size=[512, 512, 512], window=None, downsample=1, thickness_range=None, combine_left_right=False,
                 data_group_num=1, margin=1, step=1, cache_memory_num=0, parallel=False, shuffle=True, expand=1, need_negative_ratio=0.1):

        self.fold_names = np.array(fold_names, dtype=str)
        self.neg_sample_names = np.array(neg_sample, dtype=str)
        self.filenamedict = filenamedict
        self.exclude_oar = exclude_oar
        self.Num_case = len(self.fold_names)
        self.shuffle = shuffle
        self.norm_Size = np.array(Input_Size)
        self.Input_Size = np.array(Input_Size)
        self.batch_size = batch_size
        self.data_group_num=data_group_num
        self.patch_pool = []
        self.neg_patch_pool = []
        self.workers = workers
        self.Num_types = get_oar_num([i.replace('.nii.gz', '') for i in filenamedict['target']], exclude_oar, combine_left_right)
        self.case_id = 0
        self.max_z_step = Input_Size[0]//2
        self.WL = window[1]
        self.WW = window[0]
        self.combine_left_right = combine_left_right
        self.downsample = downsample
        self.thickness_range = thickness_range
        self.augmentation = augmentation
        self.margin = margin
        self.step = step
        self.cache_memory_num = cache_memory_num
        self.parallel = parallel
        self.AbnormalCaselist = []
        self.expand = expand
        self.need_negative_ratio = need_negative_ratio
        self.group_id = 0
        self.record_Input_size = self.Input_Size
        self.record_case_id = []
        self.record_group_id = 0
        self.blacklist = []

        if self.shuffle:
            random.shuffle(self.fold_names)

        # 如果需要切负类，则包含所有层
        if self.need_negative_ratio > 0:
            self.margin = None

        groupsize = max(math.ceil(self.Num_case/self.data_group_num), 1)
        self.data_group_num = max(math.ceil(self.Num_case/groupsize), 1)
        self.groupfiles = []
        for i in range(self.data_group_num):
            self.groupfiles.append(self.fold_names[i*groupsize:min((i+1)*groupsize, self.Num_case)])

        self.groupsize = len(self.groupfiles[self.group_id])

        if self.cache_memory_num is None:
            self.fact_cache_memory_num = self.groupsize
        elif self.batch_size > self.cache_memory_num:
            self.fact_cache_memory_num = self.batch_size
        else:
            self.fact_cache_memory_num = min((self.cache_memory_num // self.batch_size) * self.batch_size, self.groupsize)

        self.batch_total = self.batch_size * math.ceil(self.groupsize * self.expand / self.batch_size)



    def set_Input_Size(self, size):
        self.Input_Size = size
        if not (self.Input_Size==self.record_Input_size).all():
            del self.patch_pool
            self.patch_pool = []
            del self.neg_patch_pool
            self.neg_patch_pool = []


    def __len__(self):
        self.groupsize = len(self.groupfiles[self.group_id])
        if self.cache_memory_num is None:
            self.fact_cache_memory_num = self.groupsize
        elif self.batch_size > self.cache_memory_num:
            self.fact_cache_memory_num = self.batch_size
        else:
            self.fact_cache_memory_num = min((self.cache_memory_num // self.batch_size) * self.batch_size, self.groupsize)

        self.batch_total = self.batch_size * math.ceil(self.groupsize * self.expand / self.batch_size)

        return self.batch_total

    def get_procee(self):
        return self.case_id

    def get_Target_Name(self):
        if self.combine_left_right:
            self.targetfilelist = []
            for i in self.filenamedict["target"]:
                if i.split(".")[0] not in self.exclude_oar:
                    if '_R' not in i:
                        self.targetfilelist.append(i.replace('_L', ''))
                else:
                    self.targetfilelist.append(i)
        else:
            self.targetfilelist = self.filenamedict["target"]

        self.targetnames = [a.split(".")[0] for a in self.targetfilelist]

        return self.targetnames

    def get_downsample(self, x):
        if self.downsample > 1:
            x = x[::self.downsample, ::self.downsample, ::self.downsample]
        return x

    def crop_resize_slice(self, image_train, mask_train, targets_train, src_thickness, margin, gpu_id=0):
        return crop_resize_slice_based_roi(image_train, mask_train, targets_train, src_thickness, self.Input_Size, self.thickness_range, (not self.combine_left_right), margin, gpu_id=gpu_id)

    def get_data_from_disk(self, foldpath):
        print(foldpath)
        ## get the OARs list
        casename = foldpath

        image, affine = read_nii(os.path.join(foldpath, self.filenamedict["src"]))
        shape = image.shape
        image = np.ascontiguousarray(np.transpose(image, [2, 1, 0]))
        image = np.uint16(image)
        src_thickness = abs(affine[2, 2])

        groupmask = np.zeros(shape, dtype=np.uint16)
        for i, maskfile in enumerate(self.filenamedict["mask"]):
            mask, affine = read_nii(os.path.join(foldpath, maskfile))
            mask = self.get_downsample(mask)
            groupmask = np.maximum(groupmask, mask)

        groupmask = np.uint16(groupmask)
        groupmask = np.ascontiguousarray(np.transpose(groupmask, [2, 1, 0]))

        targets = np.zeros_like(image, dtype=np.uint16)
        index = 1
        for a in range(0, len(self.filenamedict["target"])):
            try:
                target, affine = read_nii(os.path.join(foldpath, self.filenamedict["target"][a]))
                if self.filenamedict["target"][a].replace('.nii.gz', '') not in self.exclude_oar:
                    if '_L' in self.filenamedict["target"][a] and self.combine_left_right:
                        try:
                            target_r, affine = read_nii(
                                os.path.join(foldpath, self.filenamedict["target"][a].replace('_L', '_R')))
                        except:
                            target_r = np.zeros_like(image)

                        target = np.maximum(target, target_r)

                    elif '_R' in self.filenamedict["target"][a] and self.combine_left_right:
                        continue
                    else:
                        pass

                target = self.get_downsample(target)
                target = np.ascontiguousarray(np.transpose(target, [2, 1, 0]))
                target[target == 1] = index
                target[groupmask==0] == 0
            except:
                target = np.zeros_like(image)

            index += 1
            targets = np.maximum(targets, target)

        for a in range(0, len(self.filenamedict["negative"])):
            try:
                # 添加其他部位的器官做为负类
                target, affine = read_nii(os.path.join(foldpath, self.filenamedict["negative"][a]))
                target = self.get_downsample(target)
                target = np.ascontiguousarray(np.transpose(target, [2, 1, 0]))
                target[(targets > 0) * (targets < 255)] = 0
                targets[target == 1] = 255
            except:
                pass

        targets = targets.astype(np.uint16)

        return image, groupmask, targets, src_thickness, casename


    def preprocess(self, image, mask, targets):
        if isinstance(self.WL, list):
            WL = random.uniform(self.WL[0], self.WL[1])
        else:
            WL = self.WL
        if isinstance(self.WW, list):
            WW = random.uniform(self.WW[0], self.WW[1])
        else:
            WW = self.WW

        print('WW:{}--WL:{}'.format(WW, WL))
        image = image.astype(np.float32)
        image = image - 1024
        x = np.clip(image, WL - WW // 2, WL + WW // 2) - (WL - WW // 2)

        # 组装target
        radius = 5
        y = np.zeros([self.Num_types, self.Input_Size[0], self.Input_Size[1], self.Input_Size[2]], dtype=np.float32)
        for a in range(0, self.Num_types):
            zs, ys, xs = np.where(targets == (a + 1))
            if len(zs) > 0:
                zc, yc, xc = int(np.median(zs)), int(np.median(ys)), int(np.median(xs))
                for k in range(zc-radius-1, zc+radius+2):
                    for j in range(yc - radius-1, yc + radius + 2):
                        for i in range(xc - radius-1, xc + radius + 2):
                            if k >= 0 and k < self.Input_Size[0] and j >= 0 and j < self.Input_Size[
                                1] and i >= 0 and i < self.Input_Size[2]:
                                if np.linalg.norm(np.array([k - zc, j - yc, i - xc])) <= radius:
                                    y[a][k, j, i] = 1


        # 灰度统一
        # gray windows clip
        if random.choice([True, False]):
            x = Noise_numpy(x, mask=mask, noise_var=0.2, seed=random.randint(0, 99999999))

        if x.max() > 0:
            x = x[np.newaxis, :] / x.max()
        else:
            x = x[np.newaxis, :]
            y = y * -1.0

        del image, mask, targets
        gc.collect()
        return x, y

    def release_memory(self):
        if self.fact_cache_memory_num > 0:
            try:
                del self.src
                del self.mask
                del self.target
                del self.src_thickness
                del self.casename
                del self.patch_pool

                gc.collect()
            except:
                pass

    def load_data_to_memory(self, caseid_list, casename_list):
        self.src = {}
        self.mask = {}
        self.target = {}
        self.casename = {}
        self.src_thickness = {}
        # print('start load data from disk')
        if self.parallel:
            # print(casename_list)
            num_cores = min(multiprocessing.cpu_count(), self.workers)
            tmp = Parallel(n_jobs=num_cores, verbose=0)(delayed(self.get_data_from_disk)(
                foldpath) for foldpath in casename_list)
            for i in range(len(casename_list)):
                self.src[caseid_list[i]] = tmp[i][0]
                self.mask[caseid_list[i]] = tmp[i][1]
                self.target[caseid_list[i]] = tmp[i][2]
                self.src_thickness[caseid_list[i]] = tmp[i][3]
                self.casename[caseid_list[i]] = tmp[i][4]
        else:
            for i, foldpath in enumerate(casename_list):
                # print("read img:{}".format(foldpath))
                [image, groupmask, targets, src_thickness, casename] = self.get_data_from_disk(foldpath)
                self.src[caseid_list[i]] = image
                self.mask[caseid_list[i]] = groupmask
                self.target[caseid_list[i]] = targets
                self.src_thickness[caseid_list[i]] = src_thickness
                self.casename[caseid_list[i]] = casename


    def load_neg_patch_to_pool(self):
        if self.step is None:
            self.z_step = self.Input_Size[0]
        else:
            self.z_step = random.randint(self.step // 2 + 1, self.step + 1)

        if len(self.neg_sample_names) > 0 and len(self.neg_patch_pool) == 0:
            num_neg_samples = self.groupsize
            print('load neg samples: {}'.format(num_neg_samples))
            src_list = []
            mask_list = []
            target_list = []
            src_thickness_list = []
            neg_case_name_list = []
            casename_list = [random.choice(self.neg_sample_names) for i in range(num_neg_samples)]
            if self.parallel:
                num_cores = min(multiprocessing.cpu_count(), 4)
                tmp = Parallel(n_jobs=num_cores, verbose=0)(delayed(self.get_data_from_disk)(
                    foldpath) for foldpath in casename_list)
                for i in range(len(casename_list)):
                    src_list.append(tmp[i][0])
                    mask_list.append(tmp[i][1])
                    target_list.append(tmp[i][2])
                    src_thickness_list.append(tmp[i][3])
                    neg_case_name_list.append(tmp[i][4])
            else:
                for i, foldpath in enumerate(casename_list):
                    # print("read img:{}".format(foldpath))
                    [image, groupmask, targets, src_thickness, casename] = self.get_data_from_disk(foldpath)
                    src_list.append(image)
                    mask_list.append(groupmask)
                    target_list.append(targets)
                    src_thickness_list.append(src_thickness)
                    neg_case_name_list.append(casename)

            for i in range(num_neg_samples):
                casename = neg_case_name_list[i]
                image, mask, targets, src_thickness = src_list[i], mask_list[i], target_list[i], src_thickness_list[i]
                [image, mask, targets, dst_thickness] = self.crop_resize_slice(image, mask, targets, src_thickness,
                                                                               margin=self.margin, gpu_id=0)
                print("neg dst_thickness:{}".format(dst_thickness))
                num = image.shape[0] // (self.z_step) + 1
                _, neg_patch_dict = self.split_slices(image, mask, targets, casename, num, self.Input_Size[0], need_negative_ratio=1.0)
                for key in neg_patch_dict.keys():
                    self.neg_patch_pool += neg_patch_dict[key]

            del src_list, mask_list, target_list
            gc.collect()


    def load_patch_to_pool(self, case_id):
        case_id = case_id % self.groupsize
        if (case_id not in self.record_case_id and case_id not in self.blacklist) or self.record_group_id != self.group_id\
                or (not (self.record_Input_size[1::]==self.Input_Size[1::]).all()):
            caseid_list = []
            casename_list = []
            i = 0
            j =0
            while i < self.fact_cache_memory_num:
                if (case_id + j)%self.groupsize not in self.blacklist:
                    caseid_list.append((case_id + j)%self.groupsize)
                    casename_list.append(self.groupfiles[self.group_id][(case_id + j) % self.groupsize])
                    i += 1

                j+=1

            self.record_case_id = caseid_list
            self.record_group_id = self.group_id
            self.release_memory()
            self.patch_pool = []
            self.load_data_to_memory(caseid_list, casename_list)
            image, mask, targets, src_thickness, casename = self.src[caseid_list[0]], self.mask[caseid_list[0]], self.target[caseid_list[0]], self.src_thickness[caseid_list[0]], self.casename[caseid_list[0]]
        elif case_id in self.blacklist:
            return
        else:
            image, mask, targets, src_thickness, casename = self.src[case_id], self.mask[case_id], self.target[case_id], self.src_thickness[case_id], self.casename[case_id]

        if targets[targets<255].max() == 0 and self.need_negative_ratio == 0:
            self.blacklist.append(case_id)
            return

        [image, mask, targets, dst_thickness] = self.crop_resize_slice(image, mask, targets, src_thickness, margin=self.margin, gpu_id=0)
        print("dst_thickness:{}".format(dst_thickness))
        if self.step is None:
            self.z_step = max(self.Input_Size[0]//2, 1)
        else:
            self.z_step = max(random.randint(self.step//2,  self.step), 1)

        zs, ys, xs = np.where((targets > 0) * (targets < 255))
        if len(zs) == 0:
            zs = [0, image.shape[0]]
        num = ( max(zs) - min(zs) ) // (self.z_step) + 1
        num = math.ceil(num * (1.0 + self.need_negative_ratio))
        print(casename)
        pos_patch_dict, neg_patch_dict = self.split_slices(image, mask, targets, casename, num, self.Input_Size[0], need_negative_ratio=self.need_negative_ratio)
        # 打乱数据类别顺序
        keys = list(pos_patch_dict.keys())
        random.shuffle(keys)
        fact_num = len(pos_patch_dict[keys[0]])
        for i in range(fact_num):
            for key in keys:
                if i < len(pos_patch_dict[key]):
                    self.patch_pool.append(pos_patch_dict[key][i])

        for key in keys:
            self.neg_patch_pool += neg_patch_dict[key]
        random.shuffle(self.neg_patch_pool)


    def split_slices(self, image, mask, targets, casename, num, slice_len, need_negative_ratio=0):
        D, H, W = image.shape
        pos_patch_dict = {}
        neg_patch_dict = {}

        positive_num = max(int(num * (1 - need_negative_ratio)), 1)
        negative_num = max(num - positive_num, 0)
        if targets.max() == 255:
            zs, ys, xs = np.where(targets == 255)
            zs = list(set(zs))
            zs_n = [i for i in zs if (i - self.Input_Size[0] // 2) in zs and (i + self.Input_Size[0] // 2) in zs]
            if len(zs_n) == 0:
                zs_n = []
        else:
            zs_n = []


        for i in range(1, self.Num_types+1):
            pos_patch_dict[i] = []
            zs, ys, xs = np.where(targets == i)
            zs = list(set(zs))
            if len(zs) > 0:
                if positive_num > 1:
                    midzs = np.linspace(min(zs), max(zs)+1, num=positive_num)
                elif positive_num == 1:
                    midzs = [(max(zs) + min(zs))//2]
                else:
                    raise Exception('num is zero')
                # 添加正类
                for midz in midzs:
                    midz = int(midz)
                    st = max(0, midz-slice_len//2)
                    ed = min(st + slice_len, D)
                    image_patch = np.zeros([slice_len, H, W], dtype=image.dtype)
                    mask_patch = np.zeros([slice_len, H, W], dtype=mask.dtype)
                    targets_patch = np.zeros([slice_len, H, W], dtype=targets.dtype)

                    t1 = slice_len//2 - (ed - st)//2
                    t2 = t1 + (ed - st)
                    image_patch[t1:t2] = image[st:ed]
                    mask_patch[t1:t2] = mask[st:ed]
                    targets_patch[t1:t2] = targets[st:ed]
                    pos_patch_dict[i].append((image_patch, mask_patch, targets_patch, casename))
            else:
                zs = None
                # negative_num = num

            # 添加负类
            midzs = []
            if zs is not None:
                if min(zs) > slice_len // 2:
                    midzs += list(range(0, min(zs) - slice_len // 2))
                if max(zs) + slice_len // 2 < D:
                    midzs += list(range(max(zs) + slice_len // 2, D))
            else:  # 没有正类器官
                # 有负累器官
                if targets.max() == 255 and len(zs_n) > 0:
                    midzs += zs_n

            # 如果无法切出负累，随机切以保持样本数目为固定数目
            if len(midzs) == 0:
                midzs = list(range(0, D))

            neg_patch_dict[i] = []
            for j in range(negative_num):
                midz = int(random.choice(midzs))
                st = max(0, midz - slice_len // 2)
                ed = min(st + slice_len, D)
                image_patch = np.zeros([slice_len, H, W], dtype=image.dtype)
                mask_patch = np.zeros([slice_len, H, W], dtype=mask.dtype)
                targets_patch = np.zeros([slice_len, H, W], dtype=targets.dtype)

                t1 = slice_len // 2 - (ed - st) // 2
                t2 = t1 + (ed - st)
                image_patch[t1:t2] = image[st:ed]
                mask_patch[t1:t2] = mask[st:ed]
                targets_patch[t1:t2] = targets[st:ed]
                neg_patch_dict[i].append((image_patch, mask_patch, targets_patch, casename))

            random.shuffle(pos_patch_dict[i])
            random.shuffle(neg_patch_dict[i])


        return pos_patch_dict, neg_patch_dict


    def load_batch_data(self):

        if len(self.neg_sample_names) > 0 and self.need_negative_ratio > 0:
            self.load_neg_patch_to_pool()

        iteration = 0
        while len(self.patch_pool) < self.batch_size:
            self.load_patch_to_pool(self.case_id)
            self.case_id += 1
            iteration += 1
            if iteration > 1000:
                raise Exception("bad ieteration during to zero data")


        xs = []
        ys = []
        casenames = []
        for i in range(self.batch_size):
            if len(self.neg_patch_pool) > 0:
                if random.uniform(0, 1) > self.need_negative_ratio:
                    [image_patch, mask_patch, targets_patch, casename] = self.patch_pool.pop(0)
                else:
                    [image_patch, mask_patch, targets_patch, casename] = random.choice(self.neg_patch_pool)
            else:
                [image_patch, mask_patch, targets_patch, casename] = self.patch_pool.pop(0)

            [x, y] = self.preprocess(image_patch, mask_patch, targets_patch)
            #[x, y] = self.project(image_patch, mask_patch, targets_patch)

            xs.append(x[np.newaxis,:])
            ys.append(y[np.newaxis,:])
            casenames.append(casename)

        # 合并结果
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        return xs, ys, casenames

    def __iter__(self):
        return self

    def __next__(self):
        if self.case_id >= self.batch_total:
            self.case_id = 0
            self.group_id += 1
            self.group_id = self.group_id % self.data_group_num
            self.record_Input_size = self.Input_Size
            if self.group_id != self.record_group_id:
                self.blacklist = []
                self.__len__()
            del self.neg_patch_pool
            self.neg_patch_pool = []
            raise StopIteration()
        else:
            [xs, ys, casenames] = self.load_batch_data()
            # print3Dnumpy("./test/{}".format(self.case_id), "x", xs[0,0,:,:,:])
            # print3Dnumpy("./test/{}".format(self.case_id), "y", ys[0][1::].sum(axis=0))
            return numpy_to_tensor(xs), numpy_to_tensor(ys), casenames


def get_fold_list(pathlist):
    if isinstance(pathlist, list):
        fold_names = []
        for datapath in pathlist:
            fold_names += glob.glob(datapath + '/*')
            print(datapath)
    else:
        fold_names = glob.glob(pathlist + '/*')

    return fold_names




def Generate_Train_Valid_Datasets(trainpath, validpath=None, neg_sample_path=[], filenamedict=None, exclude_oar=[], batch_size=1, workers=1,
                                  augmentation=None,Input_Size=[512, 512, 512], window=None, downsample=1, thickness_range=None, combine_left_right=False,
                                  data_group_num=1, cache_memory_num=0, shuffle=True, parallel=False, val_idx=None,
                                  val_split=0.01, expand=1, need_negative_ratio=0):
    print('Positive Sample')
    train_folder_list = get_fold_list(trainpath)
    print('Negtive Sample')
    neg_folder_list = get_fold_list(neg_sample_path)

    Num_Patient = len(train_folder_list)
    try:
        print('valid Sample')
        validlist =  get_fold_list(validpath)
        validlist = [temp for temp in validlist if os.path.isdir(temp)]
        trainlist = [temp for temp in train_folder_list if os.path.isdir(temp)]
        neg_folder_list = [temp for temp in neg_folder_list if os.path.isdir(temp)]
    except:
        # 劈分训练集和验证集
        try:
            validlist = read_list_from_txt(val_idx)
            if len(set(validlist).difference(set(train_folder_list))) > 0:
                raise Exception(val_idx + "is not right")

            trainlist = list(set(train_folder_list) - set(validlist))
            if shuffle:
                random.shuffle(trainlist)
            print("load the ready validaset")
        except:
            Num_val = math.ceil(Num_Patient * val_split)
            Num_Train = Num_Patient - Num_val
            if shuffle:
                random.shuffle(train_folder_list)
            trainlist = train_folder_list[:Num_Train]
            validlist = train_folder_list[Num_Train:]
            # 保存验证集
            write_list_to_txt(val_idx, validlist)
            print("generate new validaset")


    train_dataset = Normal_Dataset(fold_names=trainlist, neg_sample=neg_folder_list,filenamedict=filenamedict, exclude_oar=exclude_oar, batch_size=batch_size, workers=workers,
                   augmentation=augmentation, Input_Size=Input_Size, window=window, downsample=downsample, thickness_range=thickness_range,
                   combine_left_right=combine_left_right, data_group_num=data_group_num, margin=None, step=Input_Size[0]//2, cache_memory_num=cache_memory_num, parallel=parallel,
                   shuffle=True, expand=expand, need_negative_ratio=need_negative_ratio)

    valid_dataset = Normal_Dataset(fold_names=validlist, neg_sample=[], filenamedict=filenamedict, exclude_oar=exclude_oar, batch_size=batch_size, workers=workers,
                                   augmentation=augmentation, Input_Size=Input_Size,window=window, downsample=downsample, thickness_range=thickness_range,
                                   combine_left_right=combine_left_right, data_group_num=1, margin=Input_Size[0]//2, step=None, cache_memory_num=cache_memory_num, parallel=True, shuffle=False, expand=1, need_negative_ratio=0)

    return train_dataset, valid_dataset





