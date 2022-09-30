import SimpleITK as sitk
import numpy as np
import os

class Dcm_improcessing_sitk(object):
    def __init__(self):
        pass

    def finddcmseries(self, dcm_directory):
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_directory)
        dcmseriesfiles_path_list = []
        if not series_IDs:
            print("ERROR: given directory \"" + dcm_directory + "\" does not contain a DICOM series.")
        else:
            for series_ID in series_IDs:
                series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_directory, series_ID)
                if len(series_file_names) <= 1:
                    continue
                else:
                    dcmseriesfiles_path_list.append(list(series_file_names))

        return dcmseriesfiles_path_list

    def readdcmseries(self, series_file_names):
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        image_info = series_reader.Execute()
        image_array = np.array(sitk.GetArrayFromImage(image_info), dtype='int16')

        return image_array, image_info

    def loadmulitidcmseries(self, dcm_directory):
        dcmseriesfiles_path_list = self.finddcmseries(dcm_directory)
        image_array_list = []
        image_info_list = []
        if len(dcmseriesfiles_path_list) > 0:
            for seriesfiles in dcmseriesfiles_path_list:
                [image_array, image_info] = self.readdcmseries(seriesfiles)
                image_array_list.append(image_array)
                image_info_list.append(image_info)
        else:
            niipath = os.path.join(dcm_directory, 'image.nii.gz')
            if os.path.exists(niipath):
                image_info = sitk.ReadImage(niipath)
                image_array = np.array(sitk.GetArrayFromImage(image_info), dtype='int16')
                image_array_list.append(image_array)
                image_info_list.append(image_info)
                dcmseriesfiles_path_list = [niipath]

        return image_array_list, image_info_list, dcmseriesfiles_path_list


    def read_nii(self, niipath, dtype='int16'):
        image_sitk = sitk.ReadImage(niipath)
        image_array = np.array(sitk.GetArrayFromImage(image_sitk), dtype=dtype)
        return image_sitk, image_array

    def save_array_to_sitk_by_referce(self, image_array, image_sitk):
        new_image_info = sitk.GetImageFromArray(image_array)
        new_image_info.SetDirection(image_sitk.GetDirection())
        new_image_info.SetOrigin(image_sitk.GetOrigin())
        new_image_info.SetSpacing(image_sitk.GetSpacing())

        return new_image_info

    def save_array_to_sitk(self, image_array, direction, origin, spacing):
        new_image_info = sitk.GetImageFromArray(image_array)
        new_image_info.SetDirection(direction)
        new_image_info.SetOrigin(origin)
        new_image_info.SetSpacing(spacing)

        return new_image_info


    def get_image_info(self, image_info):
        pixelspace = image_info.GetSpacing()
        origin = image_info.GetOrigin()
        size = image_info.GetSize()
        direction = image_info.GetDirection()
        return origin, direction, pixelspace, size


    def crop(self, image_info, leftupon, rightdown):
        croper = sitk.CropImageFilter()
        size = image_info.GetSize()
        croper.SetLowerBoundaryCropSize(np.array(leftupon, dtype=np.uint16).tolist())
        croper.SetUpperBoundaryCropSize((np.array(size, dtype=np.uint16) - np.array(rightdown, dtype=np.uint16)).tolist())
        new_crop_info = croper.Execute(image_info)
        new_crop_array = sitk.GetArrayFromImage(new_crop_info)

        return new_crop_array, new_crop_info

    def crop_array_on_refernce(self, image_array, image_info, refernce_info):
        croper = sitk.CropImageFilter()

        src_sitk = self.save_array_to_sitk_by_referce(image_array, image_info)
        roi_size = np.array(refernce_info.GetSize()) * np.array(refernce_info.GetSpacing()) / np.array(image_info.GetSpacing())
        roi_size = roi_size.astype(np.int)
        leftupon = (np.array(refernce_info.GetOrigin()) - np.array(image_info.GetOrigin())) / np.array(image_info.GetSpacing())
        leftupon = leftupon.astype(np.int)

        rightdown = np.array(image_info.GetSize()) - roi_size - leftupon

        croper.SetLowerBoundaryCropSize(leftupon.tolist())
        croper.SetUpperBoundaryCropSize(rightdown.tolist())
        crop_info = croper.Execute(src_sitk)
        crop_array = sitk.GetArrayFromImage(crop_info)

        return crop_array, crop_info


    def crop_resize_array_on_refernce(self, image_array, image_info, refernce_info, intermethod=sitk.sitkLinear):

        _, crop_info = self.crop_array_on_refernce(image_array, image_info, refernce_info)
        resize_array, resize_info = self.resize_on_refernce(crop_info, refernce_info, intermethod=intermethod)

        return resize_array, resize_info

    @staticmethod
    def array_resize(image_array, scale, intermethod=sitk.sitkLinear):
        # image_array = image_array.astype(np.int16)
        image_info = sitk.GetImageFromArray(image_array)
        pixelspace = image_info.GetSpacing()
        origin = image_info.GetOrigin()
        size = image_info.GetSize()
        direction = image_info.GetDirection()
        vmin = image_array.min()

        if not isinstance(scale, list):
            scale = [1.0, 1.0, scale]

        newpixelspace = [pixelspace[0]*scale[0], pixelspace[1]*scale[1], pixelspace[2]*scale[2]]
        newsize = [int(size[0]/scale[0]), int(size[1]/scale[1]), int(size[2]/scale[2])]

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(intermethod)
        resample.SetDefaultPixelValue(int(vmin))
        resample.SetOutputDirection(direction)
        resample.SetOutputOrigin(origin)
        resample.SetOutputSpacing(newpixelspace)
        resample.SetSize(newsize)
        new_size_info = resample.Execute(image_info)
        new_size_array = sitk.GetArrayFromImage(new_size_info)

        return new_size_array, new_size_info


    def resize(self, image_info, newdirection, neworigin, newpixelspace, newsize, intermethod=sitk.sitkLinear):
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(intermethod)  # sitk.sitkNearestNeighbor
        image = sitk.GetArrayFromImage(image_info)
        vmin = int(image.min())
        resample.SetDefaultPixelValue(vmin)
        resample.SetOutputDirection(newdirection)
        resample.SetOutputOrigin(neworigin)
        resample.SetOutputSpacing(newpixelspace)
        resample.SetSize(newsize)
        new_size_info = resample.Execute(image_info)
        new_size_array = sitk.GetArrayFromImage(new_size_info)

        return new_size_array, new_size_info

    def resize_on_refernce(self, image_info, refernce_info, intermethod=sitk.sitkLinear):
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(refernce_info)
        image = sitk.GetArrayFromImage(image_info)
        vmin = image.min()

        resample.SetDefaultPixelValue(vmin)
        resample.SetInterpolator(intermethod)
        recover_info = resample.Execute(image_info)
        recover_array = sitk.GetArrayFromImage(recover_info)

        return recover_array, recover_info

    @staticmethod
    def crop_resize_on_roisize_dstsize(self, image_info, leftupon, rightdown, dstsize):
        [crop_array, crop_info] = self.crop(image_info, leftupon, rightdown)
        [crop_origin, direction, pixelspace, crop_size] = self.get_image_info(crop_info)

        newpixelspace = (np.array(crop_size) * np.array(pixelspace) / np.array(dstsize)).tolist()
        [resize_image, resize_info] = self.resize(crop_info, direction, crop_origin, newpixelspace, dstsize)

        return resize_image, resize_info

    def array_crop_resize_on_roisize_dstsize(self, image, leftupon, rightdown, dstsize, interp=sitk.sitkLinear):
        image_info = sitk.GetImageFromArray(image)

        [crop_array, crop_info] = self.crop(image_info, leftupon, rightdown)
        [crop_origin, direction, pixelspace, crop_size] = self.get_image_info(crop_info)

        newpixelspace = (np.array(crop_size) * np.array(pixelspace) / np.array(dstsize)).tolist()
        [resize_image, resize_info] = self.resize(crop_info, direction, crop_origin, newpixelspace, np.array(dstsize).astype(np.uint32).tolist(), intermethod=interp)

        return resize_image

    #-----------------accuracy----------------------------------------------------
    def get_HausdorffDistance(self, pred_sitk, trurh_sitk):
        HausdorffDistanceComputer = sitk.HausdorffDistanceImageFilter()
        HausdorffDistanceComputer.Execute(trurh_sitk > 0.5, pred_sitk > 0.5)
        hausdorffdistance = HausdorffDistanceComputer.GetAverageHausdorffDistance()
        return hausdorffdistance

    def get_OverlapCoff(self, pred_sitk, trurh_sitk):
        DiceComputer = sitk.LabelOverlapMeasuresImageFilter()
        DiceComputer.Execute(trurh_sitk > 0.5, pred_sitk > 0.5)
        dice = DiceComputer.GetDiceCoefficient()
        jaccard = DiceComputer.GetJaccardCoefficient()
        volume_similarity = DiceComputer.GetVolumeSimilarity()
        false_positive = DiceComputer.GetFalsePositiveError()
        false_negative = DiceComputer.GetFalseNegativeError()
        return dice, jaccard, volume_similarity, false_positive, false_negative

    # -----------------image processing----------------------------------------------------
    # 开闭运算
    def OpenAndClosing(self, image_sitk, kernelsize=5):
        image_sitk = sitk.BinaryMorphologicalOpening(image_sitk, kernelsize)
        image_sitk = sitk.BinaryMorphologicalClosing(image_sitk, kernelsize)
        return image_sitk

    # 二值化
    def BinaryThreshold(self, image_sitk, lowerThreshold=0.5, upperThreshold=1, insideValue=1, outsideValue=0):

        mask_sitk = sitk.BinaryThreshold(image_sitk, lowerThreshold=lowerThreshold, upperThreshold=upperThreshold, insideValue=insideValue,
                             outsideValue=outsideValue)
        return mask_sitk

    # 非重叠区域
    def NonOverlapping(self, mask_sitk1, mask_sitk2):
        sitk_xorop = sitk.XorImageFilter()
        sitk_mask = sitk_xorop.Execute(mask_sitk1, mask_sitk2)
        return sitk_mask

    # mask 取反向
    def NonOverlapping(self, mask_sitk):
        sitk_notop = sitk.NotImageFilter()
        inverse_mask_sitk = sitk_notop.Execute(mask_sitk)
        return inverse_mask_sitk

    # 得到轮廓
    def getContour(self, mask_sitk):
        sitk_contour = sitk.BinaryContourImageFilter()
        contour = sitk_contour.Execute(mask_sitk)
        return contour

    # 填补mask中的空洞
    def MaskFillHole(self, mask_sitk):
        fill_hole = sitk.BinaryFillholeImageFilter()
        mask_sitk = fill_hole.Execute(mask_sitk)
        return mask_sitk


    # 去除小块
    def RemoveSmallConnectedCompont(self, mask_sitk, rate=0.5):
        cc = sitk.ConnectedComponent(mask_sitk)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.SetGlobalDefaultNumberOfThreads(8)
        stats.Execute(cc, mask_sitk)
        maxsize = 0
        for l in stats.GetLabels():
            size = stats.GetPhysicalSize(l)
            if maxsize < size:
                maxlabel = l
                maxsize = size
                not_remove = []
                for l in stats.GetLabels():
                    size = stats.GetPhysicalSize(l)
                    if size > maxsize * rate:
                        not_remove.append(l)
                        labelmaskimage = sitk.GetArrayFromImage(cc)
                        outmask = labelmaskimage.copy()
                        outmask[labelmaskimage != maxlabel] = 0
                        for i in range(len(not_remove)):
                            outmask[labelmaskimage == not_remove[i]] = 255
                            outmask_sitk = sitk.GetImageFromArray(outmask)
                            outmask_sitk.SetDirection(mask_sitk.GetDirection())
                            outmask_sitk.SetSpacing(mask_sitk.GetSpacing())
                            outmask_sitk.SetOrigin(mask_sitk.GetOrigin())

        return outmask_sitk

    # 去除毛刺
    def RemovePruning(self, mask_sitk):
        Pruning = sitk.BinaryPruningImageFilter()
        mask_sitk = Pruning.Execute(mask_sitk)
        return mask_sitk


