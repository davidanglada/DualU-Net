import os
import os.path as osp

import numpy as np
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import cv2
from PIL import Image
import json

from .base import BaseCellCOCO, DetectionWrapper

KI67_CLASSES = ['Positive', 'Negative', 'Stroma']

class Ki67(torchvision.datasets.CocoDetection, BaseCellCOCO):
    def __init__(self, root, fold, transforms=None, sigma=5):
        self.root = root
        self.sigma = sigma  # Standard deviation for the Gaussian distribution

        fold = fold if isinstance(fold, str) else f'fold{fold}'
        self.fold = fold

        img_folder = osp.join(root, fold, 'images')
        ann_file = osp.join(root, fold, 'annotations.json')
        super(Ki67, self).__init__(img_folder, ann_file, transforms=transforms)
    
    @property
    def num_classes(self):
        return 3
    
    @property
    def class_names(self):
        return KI67_CLASSES
    
    def image_size(self, image_id=None, idx=None):
        return torch.tensor([1024, 1024])
    
    def __len__(self):
        return super(Ki67, self).__len__()

    def __getitem__(self, idx):
        img, tgt = super(Ki67, self).__getitem__(idx)

        # Image size should be consistent with target
        height, width = img.size

        # invalid target
        if len(tgt) > 0:
            tgt = [t for t in tgt if t['area'] > 0 and len(t['segmentation'][0]) > 4]  # remove invalid targets

        # empty target
        if len(tgt) == 0:
            # this is a dummy target that will be removed by transforms
            tgt = [dict(
                id=-1,
                image_id=idx,
                category_id=-1,
                bbox=[-1, -1, -1, -1],
                area=1024*1024,
                segmentation=[[0, 0, 0, 1024, 1024, 1024, 1024, 0]],
                centroid=[512, 512],
                iscrowd=0,
            )]

        return img, tgt
    
    def generate_gaussian_mask(self, tgt, height, width):
        """
        Generates a Gaussian mask for each centroid in the target.

        Parameters:
        - tgt: The COCO-style target annotations.
        - height, width: Dimensions of the output mask.
        
        Returns:
        - A Gaussian mask with the centroids.
        """
        mask = np.zeros((height, width), dtype=np.float32)
        
        for t in tgt:
            if 'centroid' in t:
                x, y = t['centroid']
                # Create a 2D Gaussian distribution for this centroid
                xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
                
                # Add the Gaussian to the mask
                mask += gaussian

        return torch.tensor(mask, dtype=torch.float32)
    
    def get_raw_image(self, image_id=None, idx=None):
        # get image id
        assert image_id is not None or idx is not None
        if image_id is None:
            image_id = self.ids[idx]
        # open image to RGB
        img = self._load_image(image_id)
        # convert to tensor
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)])
        img = transforms(img)
        return img

def build_ki67_dataset(cfg, split='train'):
    from .transforms import build_transforms
    root = cfg['dataset'][split]['root']
    num_classes = cfg['dataset'][split]['num_classes']
    # build transforms
    transforms = build_transforms(cfg, split, is_train = (split=='train') )
    # transforms = None
    # build dataset
    if num_classes == 1:
        dataset = DetectionWrapper(Ki67)(root, cfg['dataset'][split]['fold'],
                               transforms=transforms)
    else:
        dataset = Ki67(root, cfg['dataset'][split]['fold'],
                               transforms=transforms)
    # wrap dataset for transforms v2
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset,
                                target_keys=('image_id', 'masks', 'boxes', 'labels'))
    return dataset

def ki67_to_coco(data_dir, fold, out_dir):
    """Converts the Ki67 dataset to COCO format.
    Args:
        data_dir (str): path to the data directory.
        fold (int): fold number.
        out_dir (str): path to the output directory.
    """
    print("Converting Ki67 to COCO format...")
    
    # paths to data
    img_path = osp.join(data_dir, f"{fold}", "images", "images.npy")
    mask_path = osp.join(data_dir, f"{fold}", "masks", "masks.npy")

    # load images and masks
    images = np.load(img_path)
    masks = np.load(mask_path)[:, :, :, :-1]  # ignore background mask

    # create output directory for images
    out_dir = osp.join(out_dir, f"fold{fold}")
    if not osp.exists(osp.join(out_dir, "images")):
        os.makedirs(osp.join(out_dir, "images"))

    ls_images, ls_annots = list(), list()
    instance_count = 1  # instance id starts from 1, it is accumulated for each image

    # iterate over images
    for idx in range(images.shape[0]):
        # prepare image name (4-digit number for filename)
        filename = "im{:04d}.png".format(idx)

        # get image
        image_i = images[idx]
        # save image (in RGB format)
        Image.fromarray(image_i.astype(np.uint8)).save(osp.join(out_dir, "images", filename))

        # prepare image json
        height, width = image_i.shape[:2]
        ls_images.append(
            dict(id=idx, file_name=filename, height=height, width=width)
        )

        # prepare masks
        mask_i = masks[idx]
        # get all annotations
        for lbl in range(mask_i.shape[-1]):
            uq_instance_ids = np.unique(mask_i[:, :, lbl])[1:]
            for instance_id in uq_instance_ids:
                # get the coordinates of pixels for the current instance
                coords = np.where(mask_i[:, :, lbl] == instance_id)
                
                # get the bounding box for the current instance
                xmin = int(np.min(coords[1]))
                ymin = int(np.min(coords[0]))
                xmax = int(np.max(coords[1]))
                ymax = int(np.max(coords[0]))

                # compute the centroid
                centroid_x = int(np.mean(coords[1]))
                centroid_y = int(np.mean(coords[0]))
                centroid = [centroid_x, centroid_y]

                # get binary mask for the object
                mask_i_bin = mask_i[:, :, lbl] == instance_id
                # get contours from binary mask
                contours, _ = cv2.findContours(
                    mask_i_bin.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                # convert contours to coco format (list of list of points)
                contour = list()
                for p in contours[0].reshape(-1, 2):
                    contour.append(int(p[0]))
                    contour.append(int(p[1]))
                contour = [contour]

                # prepare annotation dict
                ls_annots.append(
                    dict(
                        id=instance_count,
                        image_id=idx,
                        category_id=int(lbl + 1),
                        bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                        area=(xmax - xmin) * (ymax - ymin),
                        segmentation=contour,
                        centroid=centroid,  # Add the centroid here
                        iscrowd=0,
                    )
                )
                instance_count += 1

    # prepare categories json
    categories = [dict(id=k + 1, name=v) for k, v in enumerate(KI67_CLASSES)]
    
    # prepare coco format json
    coco_format_json = dict(
        images=ls_images,
        annotations=ls_annots,
        categories=categories,
    )
    
    # save coco format json
    with open(osp.join(out_dir, "annotations.json"), 'w') as f:
        json.dump(coco_format_json, f)

    print(f"COCO format annotations with centroids saved to {osp.join(out_dir, 'annotations.json')}")
