import os
import numpy as np
import SimpleITK as sitk
import pandas as pd

def IsotropicResample(arg, size_ = [], interpolator = sitk.sitkLinear):
    """
    This function isotropicaly resample volumes. It can resample to specific size or calculate the matching size for the resampled volume.
    :param arg: can be SimpleITK volume or path to nii.gz file
    :param size_: the size of the new volume the default is to calculate new volume for the
    :param interpolator:
    :return: new_iamge, isotropicaly resampled volume.
    """
    if isinstance(arg, str):
        reader = sitk.ImageFileReader()
        # The option for nii images
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(arg)
        image = reader.Execute()
    else:
        image = arg
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing]*image.GetDimension()
    if len(size_)>0:
        new_size = size_
    else:
        new_size = [int(round(osz * ospc / min_spacing)) for osz, ospc in zip(original_size, original_spacing)]


    new_iamge =  sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                         image.GetPixelID())


    return new_iamge

def save_img(image,path,title ='_' ):
    """
        This function saves the image as nii.gz file.
    :param image: 3d volume
    :param path: path to the original image.
    :param title: string to add to image path
    :return:
    """

    file_name = path.split('/')[-1].split('.')[0]
    dir_path = os.path.dirname(path)
    ending = '.nii.gz'
    writer = sitk.ImageFileWriter()
    output_name = dir_path+'/'+file_name+title+ending
    writer.SetFileName(output_name)
    writer.Execute(image)



def ResampleCases(path_dir,prefix = None):
    """
    This function recieve path to directory of CORONAL nii.gz cases .Afterward,
    each coronal file with the prifex or "COR"/"Cor"/"cor" strings isotropicaly resampled and saved in the same path_dir.
    :param path_dir: Path to directory with nii.gz files to resample.
    :param prefix: String of files that need to be resampled.
    :return:
    """
    for subdir, dirs, files in os.walk(path_dir):
        for file in files:
            reader = sitk.ImageFileReader()
            # The option for nii images
            path_copy = subdir+file
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(path_copy)
            image = reader.Execute()
            if prefix is not None:
                if prefix in file:
                    pa = sitk.PermuteAxesImageFilter()
                    pa.SetOrder([0, 2, 1])
                    img_permute = pa.Execute(image)
                    isotropic_formula_image = IsotropicResample(img_permute)
                    save_img(isotropic_formula_image, path_copy, "_isotropic")
            else:
                if "Cor" in file or "COR" in file or "cor" in file  :
                    pa = sitk.PermuteAxesImageFilter()
                    pa.SetOrder([0, 2, 1])
                    img_permute = pa.Execute(image)
                    isotropic_formula_image = IsotropicResample(img_permute)
                    save_img(isotropic_formula_image, path_copy, "_isotropic")

def CreateDateBase(Path_to_data,cor_prefix=None,ax_prefix=None,train_frac=0.8,test_frac=0.1,num_folds = 1):
    """
    This function creates csv file with the path to isotropic coronal and axial files.
    Coronal and Axial files that belong to the same case should have the same case id in the beginning of file name.
    :param Path_to_data: path to directory with the data
    :param cor_prefix: the prefix of coronal data
    :param ax_prefix: the prefix of coronal data
    :param train_frac: the fraction train in the data split train:valid:test
    :param test_frac:  the fraction test in the data split train:valid:test
    :return:
    """
    data_types_mapping = {"isotropic_coronal": [], "hr_coronal":[],"hr_axial":[]}


    list_of_files = []
    for subdir, dirs, files in os.walk(Path_to_data):
        files.sort()
        for file in files:
            if "nii.gz" in file:
                print(file)
                subdir_new = subdir
                file_prifix = file.split("_")[0]+"_"+file.split("_")[1]

                if file_prifix not in list_of_files:
                    print(file_prifix)
                    list_of_files.append(file_prifix)
                if "isotropic" in file:
                    print("Enter Iso")
                    data_types_mapping["isotropic_coronal"].append(file)


                # if len(cor_prefix)>0:
                #     if cor_prefix in file:
                #         print("Enter Cor")
                #         data_types_mapping["hr_coronal"].append(file)
                # else:
                elif "Cor" in file or "COR" in file or "cor" in file or (cor_prefix!=None and cor_prefix in file):
                        print("Enter Cor")
                        data_types_mapping["hr_coronal"].append(file)
                # if len(ax_prefix)>0:
                #     if ax_prefix in file:
                #         print("Enter Ax")
                #         data_types_mapping["hr_axial"].append(file)
                else:
                    if "Ax" in file or "AX" in file or "ax" in file or (ax_prefix!=None and ax_prefix in file):
                        print("Enter Ax")
                        data_types_mapping["hr_axial"].append(file)





    print(list_of_files)
    print(data_types_mapping)
    df = pd.DataFrame(data_types_mapping, index=list_of_files)
    df = df.iloc[np.random.permutation(len(df))]
    train_size = round(len(df)*train_frac)
    test_size = round(len(df)*test_frac)
    val_size = len(df) - train_size - test_size
    for i in range(num_folds):
        test_ind = i*(test_size+val_size)+1
        if test_ind == 0:
            state = [ *['test'] * test_size,*['valid'] * val_size, *['train'] * train_size]
        else:
            if test_ind + test_size + test_size > len(df) - 1:
                diff = test_ind + test_size + test_size - ( len(df) - 1)
                state = [*['valid'] * diff,*['train'] * (test_ind - 1), *['test'] * test_size, *['valid'] * (val_size - diff),
                         *['train'] * (train_size - test_ind)]
            else:
                state = [*['train'] * (test_ind-1),*['test'] * test_size,*['valid'] * val_size,*['train'] * (train_size-test_ind+1)]
        df[str(i+1)] =state


    df.to_csv(Path_to_data + "DB.csv")
