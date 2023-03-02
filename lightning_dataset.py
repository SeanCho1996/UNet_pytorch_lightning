# %%
import os
from collections.abc import Sequence
from glob import glob
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torchvision.transforms import (Compose, InterpolationMode, Normalize,
                                    ToTensor)
from torchvision.transforms.transforms import Pad, Resize


# %%
class ResizeSquarePad(nn.Module):
    def __init__(self, target_length:int, interpolation_strategy:InterpolationMode, pad_value=0):
        super(ResizeSquarePad, self).__init__()
        if not isinstance(target_length, (int, Sequence)):
            raise TypeError(
                "Size should be int or sequence. Got {}".format(type(target_length)))
        if isinstance(target_length, Sequence) and len(target_length) not in (1, 2):
            raise ValueError(
                "If size is a sequence, it should have 1 or 2 values")

        self.target_length = target_length
        self.interpolation_strategy = interpolation_strategy
        self.pad_value = pad_value

    def forward(self, img:Image.Image):
        w, h = img.size
        if w > h:
            target_size = (
                int(np.round(self.target_length * (h / w))), self.target_length)
            img = Resize(size=target_size, interpolation=self.interpolation_strategy)(img)

            total_pad = img.size[0] - img.size[1]
            half_pad = total_pad // 2
            padding = (0, half_pad, 0, total_pad - half_pad)
            return Pad(padding=padding, fill=self.pad_value)(img)
        else:
            target_size = (self.target_length, int(
                np.round(self.target_length * (w / h))))
            img = Resize(size=target_size, interpolation=self.interpolation_strategy)(img)

            total_pad = img.size[1] - img.size[0]
            half_pad = total_pad // 2
            padding = (half_pad, 0, total_pad - half_pad, 0)
            return Pad(padding=padding, fill=self.pad_value)(img)
   
# %%
class SegDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.transform_img = Compose([
            ResizeSquarePad(target_length=512, interpolation_strategy=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.transform_mask = Compose([
            ResizeSquarePad(512, InterpolationMode.NEAREST)
        ])

        self.image_list = []

        for i in range(len(image_list)):
            self.image_list.append((image_list[i], label_list[i]))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path, label_path = self.image_list[idx]
        img = Image.open(img_path)
        mask = Image.open(label_path)
        try:
            img = self.transform_img(img)
            mask = self.transform_mask(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64) # mask transform does not contain to_tensor function
        except Exception as e:
            print(img_path)
            print(e)

        return img, mask
    
# %%
class PredSegDataset(Dataset):
    def __init__(self, img_list):
        super(PredSegDataset, self).__init__()

        self.img_list = img_list
        self.transforms = Compose([
            ResizeSquarePad(target_length=512, interpolation_strategy=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    
    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int):
        img = self.img_list[index]
        img = self.transforms(img)
        return img
    
# %%
class SegDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path=f"./PNG", test_img: Path=None, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.data_dir = data_dir
        self.test_img = test_img
        self.args = kwargs
    
    def prepare_data(self):
        self.images = self.image_fetch()

    def setup(self, stage:str="train"):
        """
        Downloads the data, parse it and split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        if stage == "train":
            self.train_data = SegDataset(self.images[0], self.images[1])
            self.val_data = SegDataset(self.images[2], self.images[3])

        if stage == "infer":
            ori_image = Image.open("/home/zhaozixiao/projects/MLFlow/PNG/image/1.png")
            self.img_size=[ori_image.size]
            self.infer_data = PredSegDataset([ori_image])


    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        train_loader = DataLoader(self.train_data,
                    shuffle=RandomSampler(self.train_data),
                    batch_size=self.args["batch_size"],
                    num_workers=self.args["num_workers"])
        return train_loader


    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        val_loader = DataLoader(self.val_data,
                    shuffle=False,
                    batch_size=self.args["batch_size"],
                    num_workers=self.args["num_workers"])
        return val_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        test_loader = DataLoader(self.val_data,
                    shuffle=False, 
                    batch_size=self.args["batch_size"],
                    num_workers=self.args["num_workers"]) 
        return test_loader
    
    def image_fetch(self, split_rate:float=0.9):
        image_train = []
        label_train = []
        image_val = []
        label_val = []

        # get images and gt folders
        img_folder = os.path.join(self.data_dir, f"image")
        gt_folder = os.path.join(self.data_dir, f"GT")

        # load images from their folder
        img_list = glob(os.path.join(img_folder, "*.png"))
        gt_list = []
        
        # load corresponding gt
        for i in img_list[:]:
            img_name = os.path.basename(i)
            gt_path = os.path.join(gt_folder, img_name)
            if os.path.exists(gt_path):
                gt_list.append(gt_path)
            else:
                img_list.remove(i)
        print(f"total train images: {len(img_list)}")

        # split train/val
        split = round(len(img_list) * split_rate)
        image_train += img_list[:split]
        label_train += gt_list[:split]
        image_val += img_list[split:]
        label_val += gt_list[split:]

        return (image_train, label_train, image_val, label_val)

# %%
if __name__ == "__main__":
    sd = SegDataModule("./PNG")
    sd.prepare_data()
    sd.setup(stage="train")

    print("xxccc")