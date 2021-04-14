import pandas as pd
import torch
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Resize, HorizontalFlip, RandomBrightnessContrast, OneOf, Blur, MotionBlur, IAAAdditiveGaussianNoise, ShiftScaleRotate
from albumentations.pytorch import ToTensor
from albumentations import Lambda, Rotate
import numpy as np
import random
from data.utils import transform, rand_bbox, test_time_aug


def create_loader(label_path, cfg, mode='train'):
    """Create dataloader

    Args:
        label_path (str): .csv path contain labels.
        cfg: contain configuration.
        mode (str, optional): train/val/test mode. Defaults to 'train'.

    Returns:
        (torch.utils.data.DataLoader): dataloader
    """
    transforms_aug = Compose([
        HorizontalFlip(),
        # RandomBrightnessContrast(
        #     always_apply=True, brightness_limit=0.2, contrast_limit=0.2),
        # OneOf([Blur(blur_limit=2, p=0.6), MotionBlur(blur_limit=3, p=0.6)], p=0.6),
        # IAAAdditiveGaussianNoise(scale=(0.01*255, 0.03*255), p=0.6),
        ShiftScaleRotate(0.05, 0.05, 5, always_apply=True),
    ])
    if mode != 'train' and cfg.tta:
        transforms_aug = test_time_aug

    pediatric_dataset = Pediatric_dataset(label_path, mode=mode, cfg=cfg,
                                          transforms=transforms_aug)

    if mode == 'train':
        loader = DataLoader(pediatric_dataset,  # collate_fn=collator,
                            batch_size=cfg.batch_size, shuffle=True, num_workers=4
                            )
    else:
        loader = DataLoader(pediatric_dataset,
                            batch_size=cfg.batch_size, shuffle=False, num_workers=4
                            )
    return loader


class Pediatric_dataset(Dataset):
    def __init__(self, label_path, cfg, mode='train', transforms=None):
        """Pediatric dataset

        Args:
            label_path (str): .csv path contain labels.
            cfg: contain configuration.
            mode (str, optional): train/val/test mode. Defaults to 'train'.
            transforms (optional): augmentations. Defaults to None.
        """
        self.data_dir = cfg.data_dir
        self.type = cfg.type
        self.cfg = cfg
        self.df = pd.read_csv(label_path)

        if self.type == 'pediatric':
            self.disease_classes = ["Other opacity", "Reticulonodular opacity", "Peribronchovascular interstitial opacity", "Diffuse aveolar opacity", "Lung hyperinflation",
                                    "Consolidation", "Bronchial thickening", "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Pneumonia"]
        elif self.type == 'chexmic':
            self.disease_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                                    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        else:
            self.disease_classes = ["No finding"]
        
        if self.type == 'chexmic':
            self.img_ids = self.df.Path.unique()
        else:
            self.img_ids = self.df.image_id.unique()

        self.transforms = transforms
        self.mode = mode
        self.n_data = len(self.img_ids)
        print(f"total images in {mode} set: {self.n_data}")

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):

        if self.mode == 'train':
            if self.cfg.aug_type == 'mixup':
                image, label = self.load_mixup_image_and_label(idx)
            elif self.cfg.aug_type == 'cutmix':
                image, label = self.load_cutmix_image_and_label(idx, 4)
            else:
                image, label = self.get_aug_image(idx)
        else:
            image, label = self.get_aug_image(idx)

        return image, label

    def get_aug_image(self, idx):
        """
        Load image and do augmentation
        """
        image, label = self.load_image_and_label(idx)
        if self.mode != 'train' and self.cfg.tta:
            image_list = self.transforms(image)
            image_list = [torch.Tensor(
                transform(image, self.cfg)).float() for image in image_list]
            image = torch.stack(image_list, dim=0)
        else:
            if self.mode == 'train':
                image = self.transforms(image=image)['image']
            image = transform(image, self.cfg)
            image = torch.Tensor(image).float()

        return image, label

    def load_image_and_label(self, idx):
        """
        Load image and its label from index
        """
        img_id = self.img_ids[idx]
        if self.type == 'chexmic':
            img_path = os.path.join(self.data_dir, img_id)
        else:
            img_path = os.path.join(
                self.data_dir, img_id+'.jpg')
        image = cv2.imread(img_path, 0)
        if self.type == 'pediatric':
            label = self.df.iloc[idx].values
            label = label[1:]
            label = np.array(label).astype(np.float32)
        elif self.type == 'chexmic':
            self.df = self.df.fillna(0)
            label = self.df.iloc[idx].values
            if self.mode == 'train':
                label = label[1:]
            else:
                label = label[5:]
            label = [random.uniform(self.smooth_range[0], self.smooth_range[1])
                     if x == -1.0 else x for x in label]
            label = np.array(label).astype(np.float32)
        else:
            label = torch.Tensor([self.df['No finding'][idx]])

        return image, label

    def load_mixup_image_and_label(self, index):
        """
        Implementation of mixup augmentation
        """
        if self.cfg.beta > 0:
            lam = np.random.beta(self.cfg.beta, self.cfg.beta)
        else:
            lam = 1
        image, label = self.get_aug_image(index)
        rand_index = random.choice(range(self.n_data))
        r_image, r_label = self.get_aug_image(rand_index)

        return lam*image+(1-lam)*r_image, lam*label+(1-lam)*r_label

    def load_cutmix_image_and_label(self, idx, num_mix):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        image, label = self.get_aug_image(idx)
        for _ in range(num_mix):
            r = np.random.rand(1)
            if self.cfg.beta <= 0 or r > self.cfg.cutmix_prob:
                continue

            lam = np.random.beta(self.cfg.beta, self.cfg.beta)
            rand_index = random.choice(range(self.n_data))

            image_rand, label_rand = self.get_aug_image(rand_index)

            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            image[:, bbx1:bbx2, bby1:bby2] = image_rand[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                       (image.size()[-1] * image.size()[-2]))
            label = label * lam + label_rand * (1. - lam)

        return image, label