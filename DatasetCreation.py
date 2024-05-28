import numpy as np
import SimpleITK as sitk
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,IterableDataset
import random
from itertools import permutations,product
import glob
import pandas as pd
import math


def flip_image(original_image,ds_image,state = 0):
    """
    This function receives original_image,ds_image and flips them according to state. In addition the patches are also rotates.
    :param original_image:
    :param ds_image:
    :param state:
    :return:
    """

    # states - 1: NO flif, 2: H flip, 3: V flip, 4: H+V flip
    #state = np.random.randint(low=1, high=5)
    if state == 0:
        state = np.random.randint(1,5)
    if state == 1:
        return original_image, ds_image
    if state == 2 or state ==4:
        original_image = torch.flip(original_image,dims=[2])
        ds_image = torch.flip(ds_image,dims=[2])
    if state == 3 or state == 4:
        original_image = torch.flip(original_image, dims=[1])
        ds_image = torch.flip(ds_image,dims=[1])
    #rotate image
    rot_state = np.random.randint(1,5)
    original_image_dim = original_image.dim()
    ds_image_dim = ds_image.dim()
    original_image = torch.rot90(original_image, rot_state, [original_image_dim - 2, original_image_dim -1])
    ds_image = torch.rot90(ds_image, rot_state, [ds_image_dim - 2, ds_image_dim -1])
    return original_image,ds_image


def non_zero_slice(vol,axis = 0):
    """
    This function receives a volume and return the first and last slices with non zero values on spacific direction.
    :param vol: 3d array
    :param axis: direction to check (0-axial,1-coronal,2 segittal)
    :return: first_slice first non zero slice,last_slice last non zero slice
    """

    non_zero = np.nonzero(np.array(vol))
    first_slice = np.min(non_zero[axis])
    last_slice = np.max(non_zero[axis])
    return first_slice,last_slice

def load_nifti_image(path,file_type) -> object:
    """
    This function receives path to file and file type and than loads the file to 3d volume with SimpleITK.
    The function convert it to torch.tensor and normalize its values between 0 to 1.
    :param path: path to filr
    :param file_type: file type nifti or diicom
    :return: tensor  -  normalized tensor
    """

    reader = sitk.ImageFileReader()
    # The option for nii images
    if file_type is None or file_type == 'nifti':
        reader.SetImageIO("NiftiImageIO")
    else:
        reader.SetImageIO('GDCMImageIO')
    reader.SetFileName(path)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)
    tensor = torch.from_numpy(image_array)
    # if mode == 1:
    #     tensor = tensor[2:-1,:,:]
    #Normlize the value of pixsels [0,1]
    tensor = (tensor - torch.min(tensor))/ (torch.max(tensor)- torch.min(tensor))
    return tensor

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(np.floor((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
    print("worker ID-{},Dataset_start-{},Dataset_end-{},per_worker-{}".format(worker_id,dataset.start,dataset.end,per_worker))


def return_list_of_valuble_slices(file_path,num_of_consecutive_slices = 3,axis=0):
    """
    This function return list of all the non-zero slices in the volume in the following form
    [file_path,slice_number in the volume, augmentation state [1], [slice_width, slice_highet]]
    :param file_path: full path to the file
    :param num_of_consecutive_slices: Number of consecutive slices to use.
    :param axis: which direction of slices to use.
    :return:
    """
    temp_file = load_nifti_image(file_path,  'nifti')
    first_slice, last_slice = non_zero_slice(temp_file)
    # if temp_file.shape[1] % 2 !=0:
    #     temp_file = temp_file[:,:-1,:]
    # if temp_file.shape[2] % 2 !=0:
    #     temp_file = temp_file[:,:,:-1]
    if first_slice > num_of_consecutive_slices:
        first_slice = first_slice - num_of_consecutive_slices
    else:
        first_slice =int(num_of_consecutive_slices/2)
    if last_slice < temp_file.shape[axis] - num_of_consecutive_slices:
        last_slice = last_slice + num_of_consecutive_slices
    else:
        last_slice = temp_file.shape[axis] -1 - int(num_of_consecutive_slices/2)
    diff = last_slice - first_slice + 1
    ind_range = range(first_slice, last_slice)
    list_of_slices = list(zip([file_path] * len(ind_range), ind_range,  [1]* len(ind_range),[(temp_file.shape[1],temp_file.shape[2])]* len(ind_range)))

    return list_of_slices

def assign_patch(item, value):
    """
    This function concatenates value to item
    :param item:
    :param value:
    :return:
    """
    item = list(item)
    item.append((int(value[0]),int(value[1])))
    return tuple(item)



def calculate_overlap(image_size,patch_size):
    """
    THis function calculates the optimal overlap between the patches for a given image size.
    :param image_size: the size of image
    :param patch_size: The size of the patches
    :return: possible_overlap[ind_min] - optimal overlap.
    """
    num = image_size - patch_size
    possible_overlap = np.arange(patch_size - int(patch_size/3)-3,patch_size - int(patch_size/4)+3)
    amount_of_patchs = num / possible_overlap
    toatal = np.abs(np.round(amount_of_patchs) - amount_of_patchs )
    ind_min = np.argmin(toatal)
    return possible_overlap[ind_min]


def extract_patches(list_of_slices, patch_size = 64,train = False,state = False):
    """
    This function receive list_of_slices and extract overlap patches from all the slices.
    :param list_of_slices: list of slices from return_list_of_valuble_slices
    :param patch_size: the size of overlap patches
    :param train: Bool. Is it train set or not.
    :param state: Bool. To add augmentation state or not.
    :return: coustom_slices list of extracted patches from
    """
    coustom_slices_old = list_of_slices
    new_list = []
    for i in range(len(coustom_slices_old)):
        coustom_slices_old[i] = list(coustom_slices_old[i])
        size_x , size_y = coustom_slices_old[i][3]
        overlap_x = calculate_overlap(size_x,patch_size)
        overlap_y = calculate_overlap(size_y, patch_size)


        x_start = np.arange(0, size_x - patch_size + 1, overlap_x)
        y_start = np.arange(0, size_y - patch_size + 1, overlap_y)
        try:
            if size_x - (x_start[-1] + patch_size) > (patch_size / 2):
                x_start = np.append(x_start, x_start[-1] + overlap_x)
            if size_y - (y_start[-1] + patch_size) > (patch_size / 2):
                y_start = np.append(y_start, y_start[-1] + overlap_y)
        except:
            print("got error with the file {}".format(coustom_slices_old[i]))

        patch_id = list(product(x_start, y_start))
        if x_start[-1]+patch_size != size_x or y_start[-1]+patch_size != size_y:
            coustom_slices_old[i][3] = (int(x_start[-1]+patch_size),int(y_start[-1]+patch_size))
        else:
            coustom_slices_old[i][3] = (int(size_x) , int(size_y))
        coustom_slices_old[i] = tuple(coustom_slices_old[i])
        temp_list = [coustom_slices_old[i]] * len(patch_id)
        temp_list = list(map(assign_patch, temp_list, patch_id))
        new_list += temp_list
    coustom_slices = new_list

    if train:
        if state:
            tmp_list = []
            for state in range(2, 5):
                tmp_list += [(item[0], item[1], state, item[3],item[4]) for item in coustom_slices]
            coustom_slices += tmp_list
            random.shuffle(coustom_slices)
        else:
            coustom_slices = [(item[0], item[1], 0, item[3],item[4]) for item in coustom_slices]

    return coustom_slices



def create_patch_list(list_files,max_patches,num_of_consecutive_slices,train = False,patch_size = 0,state = False):
    """
    This function receives list of_files and returns list of extracted patchs from this list.
    :param list_files: list of chosen files.
    :param max_patches: Maximum amount of patches.
    :param num_of_consecutive_slices: Number of consecutive slices
    :param train: Bool. Is the set is train or not
    :param patch_size: the size of the patch.
    :param state: Bool. To add all rotations to patch or only one of them.
    :return: coustom_slices list of patches.
    """


    coustom_slices_1 = []
    coustom_slices_2 = []
    total_1 = 0
    total_2 = 0

    for j in range(len(list_files)):

        temp_list_1 =  return_list_of_valuble_slices(list_files[j][0],num_of_consecutive_slices)
        total_1 += len(temp_list_1)
        coustom_slices_1.extend(temp_list_1)
        temp_list_2 =  return_list_of_valuble_slices(list_files[j][1],num_of_consecutive_slices)
        total_2 += len(temp_list_2)
        coustom_slices_2.extend(temp_list_2)
    random.shuffle(coustom_slices_1)
    random.shuffle(coustom_slices_2)

    if not train:

        coustom_slices = [coustom_slices_1,coustom_slices_2]

    if train:
        coustom_slices_1 = extract_patches(coustom_slices_1, patch_size,train,state)
        if len(coustom_slices_1)>max_patches:
            coustom_slices_1 = coustom_slices_1[0:max_patches]
        else:
            ratio = math.ceil(max_patches/len(coustom_slices_1))
            coustom_slices_1_extend = coustom_slices_1 * ratio
            random.shuffle(coustom_slices_1_extend)
            coustom_slices_1 = coustom_slices_1_extend[:max_patches]
        coustom_slices_2 = extract_patches(coustom_slices_2,  patch_size,train,state)
        if len(coustom_slices_2)>max_patches:
            coustom_slices_2 = coustom_slices_2[0:max_patches]
        ratio = math.ceil(len(coustom_slices_1)/len(coustom_slices_2))


        coustom_slices_2_extend = coustom_slices_2 * ratio
        random.shuffle(coustom_slices_2_extend)
        coustom_slices_2_extend = coustom_slices_2_extend[:len(coustom_slices_1)]

        coustom_slices = [((i[0], j[0]), (i[1], j[1]), (i[2], j[2]), (i[3], j[3]),(i[4], j[4])) for i, j in zip(coustom_slices_1, coustom_slices_2_extend)]
        random.shuffle(coustom_slices)


    return coustom_slices


def crop_new_size (vol,ind, new_size):
    """
    This function reshapes vol to new size
    :param vol: given volume
    :param ind: index of one of slices
    :param new_size: the desired new size
    :return: vol_cropped , vol with new size
    """
    original_size = np.array(vol[ind,:,:].shape)
    diff = (np.array(vol[ind,:,:].shape)-np.array(new_size))
    if original_size[0] <list(new_size)[0]:
        p2d = (0, 0, int(abs(diff[0])/2), abs(diff[0]) - int(abs(diff[0])/2), 0, 0)
        vol = F.pad(vol, p2d, "constant", 0)
        first_ind_cor = 0
        last_ind_cor = new_size[0]
    elif original_size[0] == list(new_size)[0] :
        first_ind_cor = 0
        last_ind_cor = new_size[0]
    else:
        vol = vol[:,int(abs(diff[0])/2):-(abs(diff[0]) - int(abs(diff[0])/2)),:]
        first_ind_cor = 0
        last_ind_cor = new_size[0]

    if original_size[1] < list(new_size)[1]:
        p2d = (int(abs(diff[1])/2), abs(diff[1]) - int(abs(diff[1])/2),0, 0, 0, 0)
        vol = F.pad(vol, p2d, "constant", 0)
        first_ind_row = 0
        last_ind_row = new_size[1]
    elif original_size[1] == list(new_size)[1] or int(diff[1]/2) == 0:
        first_ind_row = 0
        last_ind_row = new_size[1]
    else:
        vol = vol[:,:,int(abs(diff[1])/2):-(abs(diff[1]) - int(abs(diff[1])/2))]
        first_ind_row = 0
        last_ind_row = new_size[1]

    vol_cropped = vol[:,first_ind_cor:last_ind_cor,first_ind_row:last_ind_row]
    return vol_cropped


def  extract_patches_with_volume(volume, patch_size = 64):
    """

    :param volume:
    :param patch_size:
    :return:
    """

    size_x , size_y = volume.shape[1],volume.shape[2]
    final_size_x,final_size_y  = size_x , size_y
    overlap_x = calculate_overlap(size_x,patch_size)
    overlap_y = calculate_overlap(size_y, patch_size)



    x_start = np.arange(0, size_x - patch_size + 1, overlap_x)
    y_start = np.arange(0, size_y - patch_size + 1, overlap_y)

    if size_x - (x_start[-1]+patch_size) > (patch_size/2):
        x_start = np.append(x_start,x_start[-1]+overlap_x)
    if size_y - (y_start[-1]+patch_size) > (patch_size/2):
        y_start = np.append(y_start,y_start[-1]+overlap_y)
    patch_id = list(product(x_start, y_start))
    if x_start[-1]+patch_size != size_x or y_start[-1]+patch_size != size_y:
        final_size = (int(x_start[-1]+patch_size),int(y_start[-1]+patch_size))
    else:
        final_size = (int(size_x) , int(size_y))

    final_volume = crop_new_size(volume,0,final_size)

    return final_volume,patch_id

def pad_vol_patches(list_tensor):
    """
    This function
    :param list_tensor:
    :return:
    """

    vol_sizes = [elem[1][0] for elem in list_tensor]
    max_patches = max(vol_sizes)
    for i in range(len(list_tensor)):
        if list_tensor[i][1][0] >= max_patches:
            continue
        else:
            diff = max_patches - list_tensor[i][1][0]
            pad = (0, 0, 0, 0, 0, diff)
            list_tensor[i][0] = F.pad(list_tensor[i][0], pad, value=0)


    return list_tensor

def create_patch_list_and_volume_list(list_files,train = False,patch_size = 0):
    """

    :param list_files:
    :param train:
    :param patch_size:
    :return:
    """
    def split_volume_to_patches(volume,patch_size):
        """

        :param volume:
        :param patch_size:
        :return:
        """
        overlap_x = calculate_overlap(volume.size()[-2], patch_size)
        overlap_y = calculate_overlap(volume.size()[-1], patch_size)
        x_start = np.arange(0, volume.size()[-2] -patch_size + 1, overlap_x)
        y_start = np.arange(0, volume.size()[-1] -patch_size + 1, overlap_y)
        img_size_o = (volume.size()[-1],volume.size()[-2])
        if x_start[-1] + patch_size != volume.size()[-1] or y_start[-1] + patch_size != volume.size()[-2]:
            img_size_o = (x_start[-1] + patch_size, y_start[-1] + patch_size)
        volume = volume[:,:img_size_o[0],:img_size_o[1]]
        kc, kh, kw = 1, patch_size, patch_size  # kernel size
        dc, dh, dw = 1, overlap_x, overlap_y  # stride
        patches = volume.unfold(0, kc, dc).unfold(1, kh, dh).unfold(2, kw, dw)
        patches = patches.contiguous().view(volume.size()[0], -1, kh, kw)
        img_size_o = (patches.size()[1],img_size_o[0],img_size_o[1])
        return patches,img_size_o



    list_LR_tensor = []
    list_HR_tensor = []
    file_to_idx = {}

    if len(list_files)==0:
        return [[], []], {}

    for j in range(len(list_files)):
        temp_volume = load_nifti_image(list_files[j][0], 'nifti')

        temp_volume,_ = extract_patches_with_volume(temp_volume, patch_size )
        if not train:
            split_volume,img_size = split_volume_to_patches(temp_volume,patch_size)
            list_LR_tensor.append([split_volume,img_size])
        else:
            list_LR_tensor.append(temp_volume)
        file_to_idx[list_files[j][0]] = j
        temp_volume = load_nifti_image(list_files[j][1], 'nifti')
        temp_volume, _ = extract_patches_with_volume(temp_volume, patch_size)
        if not train:
            split_volume,img_size = split_volume_to_patches(temp_volume,patch_size)
            list_HR_tensor.append([split_volume,img_size])
        else:
            list_HR_tensor.append(temp_volume)

        file_to_idx[list_files[j][1]] = j
    if not train:
        list_LR_tensor = pad_vol_patches(list_LR_tensor)
        list_HR_tensor = pad_vol_patches(list_HR_tensor)

    return [list_LR_tensor, list_HR_tensor], file_to_idx

def worker_init_fn(worker_id):
    """
    Thisfunction recieves worker_id and assign part of the dataset to this worker.
    :param worker_id:
    :return:
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(np.floor((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
    print("worker ID-{},Dataset_start-{},Dataset_end-{},per_worker-{}".format(worker_id,dataset.start,dataset.end,per_worker))


def random_patch_coor(image_size,patch_size):

    x_coor = np.random.randint(image_size[0]-patch_size-1)
    y_coor = np.random.randint(image_size[1]-patch_size-1)

    return x_coor, y_coor

def split_files_list_from_db(path_to_set,max_files,max_patches,num_of_consecutive_slices=3,max_slices=None,patch_size = 0,fold = 1,state = False):
    """
    This function creates the datasets of train,validation and test from the DB file. Afterwards it creats dataloaders for them.
    The validation and test sets have seperate dataloders for the low tn high sets.
    :param path_to_set: path to the data folder.
    :param max_files: max number of iles to use from the total files in the data.
    :param max_patches: max number of patches after extracting all possible overlapping patches from all the slices of the chosen files
    :param max_slices: max amount of slices from all the chosen files
    :param patch_size: the size of the desired patch
    :param fold: number of fold in case of cross validation
    :param ddp_seed: seed number
    :param load_all: load all the files to the memory
    :param state: Bool. whetere to use 1 random rotation segmentation or to add 4 rotation to each patch
    :return:
    """


    path_to_db = path_to_set + 'DB.csv'
    df = pd.read_csv(path_to_db)

    print("path_to_db-", path_to_db)


    df[["isotropic_coronal", "hr_coronal","hr_axial"]] = path_to_set + df[["isotropic_coronal", "hr_coronal","hr_axial"]]

    if fold is not None:
        # df.loc[df['state'] == 'valid','state'] = 'train'
        # df.loc[df['group'] == fold,'state'] = 'valid'
        df['state'] = df[str(fold)]
    else:
        df['state'] = df["1"]

    if max_files is not None and max_files < len(df):
        ratio = max_files/ len(df[df['state']=='train'])
        df_train = df[df['state']=='train'].reset_index(drop = True)
        df_train = df_train.loc[0:max_files]
        df_valid = df[df['state']=='valid'].reset_index(drop = True)
        df_valid = df_valid.iloc[0:round(len(df_valid)*ratio)]
        df_test = df[df['state']=='test'].reset_index(drop = True)
        df_test = df_test.iloc[0:round(len(df_test)*ratio)]
    else:
        max_files = len(df)
        df_train = df[df['state'] == 'train']
        df_valid = df[df['state'] == 'valid'].reset_index(drop=True)
        df_test = df[df['state'] == 'test'].reset_index(drop=True)




    list_test = df_test[['isotropic_coronal','hr_axial']].values.tolist()
    list_val = df_valid[['isotropic_coronal', 'hr_axial']].values.tolist()
    list_train = df_train[['isotropic_coronal','hr_axial']].values.tolist()


    list_train_slices = create_patch_list(list_train, max_patches,num_of_consecutive_slices, train=True, patch_size=patch_size,state = state)
    #list_train_slices = create_slice_list(list_train,max_slices, bratz,True,patch_size = patch_size)
    list_test_slices = create_patch_list(list_test, max_patches,num_of_consecutive_slices,patch_size = patch_size)
    list_val_slices = create_patch_list(list_val, max_patches,num_of_consecutive_slices,patch_size = patch_size)

    print("length train_list - ",len(list_train_slices))
    print("length val_list - ", len(list_val_slices))
    print("length test_list - ", len(list_test_slices))


    list_train_volume,train_file_to_idx = create_patch_list_and_volume_list(list_train, train=True, patch_size=patch_size)
    #list_train_slices = create_slice_list(list_train,max_slices, bratz,True,patch_size = patch_size)
    list_test_volume,test_file_to_idx = create_patch_list_and_volume_list(list_test, train = False,patch_size = patch_size)
    list_val_volume,val_file_to_idx = create_patch_list_and_volume_list(list_val, train = False,patch_size = patch_size,)

    print("length train_list volume - ",len(list_train_volume))
    print("length val_list volume - ", len(list_val_volume))
    print("length test_list volume - ", len(list_test_volume))

    return list_train_slices,list_val_slices,list_test_slices,False, list_train_volume, train_file_to_idx, \
        list_test_volume, test_file_to_idx, list_val_volume, val_file_to_idx



class CustomDataset_Train(Dataset):

    def __init__(self, slices,file_list ,file_type = None, batch = None,patch_size = 0,
                use_db=False,list_volumes=None, file_to_idx=None,random_patch=True):
        """
        """
        super().__init__()
        # sitk.ProcessObject.SetGlobalWarningDisplay(False)
        self.list_slices = file_list
        self.slices = slices
        self.max_slices = len(self.list_slices)
        self.start = 0
        self.end = len(self.list_slices)
        self.file_type = file_type
        self.batch = batch
        self.patch_size = patch_size
        self.list_volumes = list_volumes
        self.file_to_idx = file_to_idx
        if patch_size > 0:
            self.use_patches = True
        else:
            self.use_patches = False
        #self.length = slices
        self.use_db = use_db

        self.random_patch = random_patch
    def __len__(self):
        return self.max_slices

    def __getitem__(self, idx):
        # Create an iterator
        lim = int(self.slices/2)

        file, ind, state, img_size, patch_id = self.list_slices[idx]
        file_idx_LR = self.file_to_idx[file[0]]
        file_idx_HR = self.file_to_idx[file[1]]


        original_tensor = self.list_volumes[1][file_idx_HR]
        down_tensor = self.list_volumes[0][file_idx_LR]
        if self.use_patches:
            x_start_o, y_start_o = patch_id[1]
            x_start_d, y_start_d = patch_id[0]
            if self.random_patch:
                x_start_o,y_start_o = random_patch_coor(list(img_size[1]), self.patch_size)
                x_start_d, y_start_d = random_patch_coor(list(img_size[0]), self.patch_size)
            original_tensor = original_tensor[:, x_start_o:x_start_o + self.patch_size,
                              y_start_o:y_start_o + self.patch_size]
            down_tensor = down_tensor[:, x_start_d:x_start_d + self.patch_size,
                          y_start_d:y_start_d + self.patch_size]
        original_tensor, down_tensor = flip_image(original_tensor, down_tensor, state[0])

        down_tensor = torch.squeeze(down_tensor[ind[0] - lim:ind[0] + lim + 1, :, :])
        # original_tensor_sq = torch.squeeze(original_tensor[ind])

        return torch.unsqueeze(down_tensor, 0).float(), \
                torch.unsqueeze(original_tensor[ind[1]], 0).float()


class CustomDataset_Test(Dataset):

    def __init__(self, slices,file_list ,lr = True, file_type = None, batch = None,patch_size = 0,
                 use_db=False,list_volumes=None, file_to_idx=None):
        """
        """
        super().__init__()

        self.list_slices = file_list
        self.slices = slices
        self.max_slices = len(self.list_slices)
        self.start = 0
        self.end = len(self.list_slices)

        self.file_type = file_type
        self.batch = batch
        self.patch_size = patch_size
        self.lr = lr
        self.list_volumes = list_volumes
        self.file_to_idx = file_to_idx
        if patch_size > 0:
            self.use_patches = True
        else:
            self.use_patches = False
        #self.length = slices
        self.use_db = use_db



    def __len__(self):
        return self.max_slices #int(np.floor(self.max_slices/self.batch))

    def __getitem__(self, idx):
        # Create an iterator
        lim = int(self.slices/2)

        file, ind, _ ,img_size = self.list_slices[idx]

        file_idx = self.file_to_idx[file]
        original_tensor = self.list_volumes[file_idx][0]
        img_size_o = self.list_volumes[file_idx][1]

        if self.lr:
            original = original_tensor[ind - lim:ind + lim + 1,:, :, :]
            print(original.shape)
            original = torch.permute(original, (1, 0, 2, 3))
            print(original.shape)
        else:
            original = original_tensor[ind, :, :, :]
            print(original.shape)

        title = file + '_slice_' + str(ind)

        if original.isnan().any():
            print("nan patches")
        return  original, img_size_o, title

