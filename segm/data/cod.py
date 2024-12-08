from pathlib import Path
import torch
from torch.utils.data import Dataset
from segm.data import utils
from PIL import Image
import cv2
from segm.config import dataset_dir
from segm.data import transform
from mmcv.utils import Config
import os
import glob
import numpy as np
from skimage import segmentation
PASCAL_CONTEXT_CONFIG_PATH = Path(__file__).parent / "config" / "cod.py"
PASCAL_CONTEXT_CATS_PATH = Path(__file__).parent / "config" / "cod.yml"


class CODDataset(Dataset):
    def __init__(self, data_root, image_size, crop_size, split, normalization, **kwargs):
        super().__init__()
        self.names, self.colors = utils.dataset_cat_description(
            PASCAL_CONTEXT_CATS_PATH
        )

        self.n_cls = 2
        self.ignore_label = 255
        self.reduce_zero_label = False
        self.split = split
        self.image_size = image_size
        self.crop_size = crop_size
        self.data_root = data_root

        config = Config.fromfile(PASCAL_CONTEXT_CONFIG_PATH)
        self.ratio = config.max_ratio
        self.normalization = utils.STATS[normalization].copy()
        self.config = self.update_default_config(config)
        data_root = data_root
        filename = split
        if filename == 'val':
            filename = 'val_COD10K_1000' #'val_COD10K' #'val_CAMO'
        self.imglist = self.get_imglist(os.path.join(data_root, filename, 'Imgs'))
        print("Total Images", len(self.imglist))
        if split == 'train':
            trans = transform.Compose([
                transform.Resize((image_size, image_size), (288,288)),  # 336
                #transform.RandomGaussianBlur(),
                #transform.RandomNoise(),
                #transform.RandomMask(),
                #transform.RandRotate([-30,30],padding=list(self.normalization['mean']), ignore_label=0),
                transform.RandomHorizontalFlip(),
                transform.RandomVerticalFlip(),
                transform.ToTensor(),
                transform.Normalize(mean=self.normalization['mean'], std=self.normalization['std'])
            ])
            
        elif split == 'val':
            trans = transform.Compose([
                transform.Resize((image_size, image_size), (288,288)), # 336
                transform.ToTensor(),
                transform.Normalize(mean=self.normalization['mean'], std=self.normalization['std'])
            ])
        else:
            trans = None
        self.transform = trans
        self.edge_transform = transform.Compose([
                transform.Resize((image_size, image_size), (288,288)),  # 336
                transform.ToTensor(),
                transform.Normalize(mean=self.normalization['mean'], std=self.normalization['std'])
            ])

    def get_imglist(self, path):
        
        img_list = glob.glob(os.path.join(path, '*.jpg'))
        return img_list

    def update_root_config(self, config):

        train_splits = ["train", "trainval"]
        if self.split in train_splits:
            config_pipeline = getattr(config, f"train_pipeline")
        else:
            config_pipeline = getattr(config, f"{self.split}_pipeline")

        img_scale = (self.ratio * self.image_size, self.image_size)
        if self.split not in train_splits:
            assert config_pipeline[1]["type"] == "MultiScaleFlipAug"
            config_pipeline = config_pipeline[1]["transforms"]
        for i, op in enumerate(config_pipeline):
            op_type = op["type"]
            if op_type == "Resize":
                op["img_scale"] = img_scale
            elif op_type == "RandomCrop":
                op["crop_size"] = (
                    self.crop_size,
                    self.crop_size,
                )
            elif op_type == "Normalize":
                op["mean"] = self.normalization["mean"]
                op["std"] = self.normalization["std"]
            elif op_type == "Pad":
                op["size"] = (self.crop_size, self.crop_size)
            config_pipeline[i] = op
        if self.split == "train":
            config.data.train.pipeline = config_pipeline
        elif self.split == "trainval":
            config.data.trainval.pipeline = config_pipeline
        elif self.split == "val":
            config.data.val.pipeline[1]["img_scale"] = img_scale
            config.data.val.pipeline[1]["transforms"] = config_pipeline
        elif self.split == "test":
            config.data.test.pipeline[1]["img_scale"] = img_scale
            config.data.test.pipeline[1]["transforms"] = config_pipeline
            config.data.test.test_mode = True
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return config

    def update_default_config(self, config):
        if self.data_root is None:
            root_dir = '../../data'
        else:
            root_dir = self.data_root
        path = Path(root_dir) / "DIS5K"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path / "DIS-TR/"
        elif self.split == "val":
            config.data.val.data_root = path / "DIS-VD/"
        elif self.split == "test":
            raise ValueError("Test split is not valid for Pascal Context dataset")
        config = self.update_root_config(config)
        return config

    def test_post_process(self, labels):
        return labels

    def get_gt_seg_maps(self):
        gt_seg_maps = {}
        for filepath in self.imglist:
            filename = filepath.split('/')[-1]
            label_path = filepath.replace('Imgs', 'GT').replace('.jpg', '.png')
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            gt_seg_maps[filename] = label / 255
        return gt_seg_maps

    def __getitem__(self, idx):
        imgpath = self.imglist[idx]
        image_np = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        org_size = image_np.shape
        label_path = imgpath.replace('Imgs', 'GT').replace('.jpg', '.png')
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        #label = cv2.resize(label, (128,128))
        if 'Edge' in label_path:
            label = cv2.dilate(label, (11,11), iterations=7)
        label[np.where(label >= 127)] = 255
        # cv2.imwrite('a.png', label)
        # exit()
        if self.transform is not None:
            image, label = self.transform(image_np, label)
            # add_adge
            np_label = label.cpu().detach().numpy()*255
            mask_sobel = np.zeros_like(np_label)
            bd = segmentation.find_boundaries(np_label, mode="inner")
            mask_sobel[bd] = [255]
            mask_sobel = cv2.dilate(mask_sobel.astype('uint8'), (9,9), iterations=3)
            mask_sobel[np.where(mask_sobel > 127)] = 255
            _, edge = self.edge_transform(image_np, mask_sobel)
            
        return {'im':image, 'segmentation':label, 'edge':edge.type_as(label), 'filename':imgpath,'ori_size':org_size}

    def __len__(self):
        return len(self.imglist)

    @property
    def unwrapped(self):
        return self

    def set_epoch(self, epoch):
        pass

    def get_diagnostics(self, logger):
        pass

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return


    
