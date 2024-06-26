# Reading the dataset

import os
from PIL import Image
import numpy as np
import torch
import pandas as pd
import ast
from torch.utils import data
from data_utils import CDDataAugmentation

"""
Dataset folder structure:
├─previous_image
├─post_image
├─label
└─list - train.txt,test.txt, val.txt
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"


label_suffix='.txt'

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def load_image_label_list_from_txt(txt_path, img_name_list):
    label_dict = {}
    with open(txt_path) as f:
        for line in f:
            img_name, label = line.strip().split()
            label_dict[img_name] = label
    return [label_dict[img_name] for img_name in img_name_list]

def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


class ImageDataset(data.Dataset):
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  #train | train_aug | val
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

         # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

        self.train_df = pd.read_csv('/home/amarachi/CheXRelFormer/data_extraction/train.csv')
        self.test_df = pd.read_csv('/home/amarachi/CheXRelFormer/data_extraction/test.csv')
        self.val_df = pd.read_csv('/home/amarachi/CheXRelFormer/data_extraction/val.csv')
        self.all_df = pd.concat([self.train_df, self.val_df, self.test_df], axis=0)
        self. all_df['concatenated_id'] = self.all_df['current_image_id'] + '_' + self.all_df['previous_image_id']+ ".jpg"
        self.A_size = len( self.img_name_list) 
    

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.filtered_img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir,self.filtered_img_name_list[index % self.A_size])

        try:
            img = np.asarray(Image.open(A_path).convert('RGB'))
        except OSError as e:
            print("Failed to load image:", A_path)
            
        try:
            img_B = np.asarray(Image.open(B_path).convert('RGB'))
        except OSError as e:
            print("Failed to load image:", B_path)

        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name =  self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir,self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        L_path = get_label_path(self.root_dir,self.img_name_list[index % self.A_size])

        with open(L_path, 'r') as f:
            label = f.read()
        label = label.strip()
        label_to_int = {'worsened': 0, 'improved': 1, 'no change': 2}
        label = label_to_int[label]

        bboxA = self.all_df.loc[self.all_df['concatenated_id'] == name, 'bbox_coord_224_object'].values[0]
        bboxB = self.all_df.loc[self.all_df['concatenated_id'] == name, 'bbox_coord_224_subject'].values[0]
        bboxA = ast.literal_eval(bboxA) 
        bboxB = ast.literal_eval(bboxB) 
       
        # xmin, ymin, xmax, ymax = bboxA
        # xminB, yminB, xmaxB, ymaxB = bboxB
        # img = img[ymin:ymax, xmin:xmax, :]
        # img_B = img_B[yminB:ymaxB, xminB:xmaxB, :]
        img = img.crop(bboxA)
        img_B = img_B.crop(bboxB)
        
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
      
        return {'name': name, 'A': img, 'B': img_B, 'L': label}
