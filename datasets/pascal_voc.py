#-*-coding:utf-8-*-
'''
Created on Oct 24,2018

@author: pengzhiliang
'''
import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
# Tqdm 是一个快速，可扩展的Python进度条，可以�?Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代�?tqdm(iterator)
from tqdm import tqdm

from torch.utils import data
from torchvision import transforms

"""
Pascal Voc 2012 & benchmark_release 数据集介绍：
详情请见：https://blog.csdn.net/iamoldpan/article/details/79196413

VOC2012数据集分�?0类，包括背景�?1类，分别如下�?
- Person: person 
- Animal: bird, cat, cow, dog, horse, sheep 
- Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train 
- Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

VOC2012中的图片并不是都用于分割，用于分割比赛的图片包含原图、图像分类分割和图像物体分割两种png图�?图像分类分割：在20种物体中，ground-turth图片上每个物体的轮廓填充都有一个特定的颜色，一�?0种颜色，比如摩托车用红色表示，人用绿色表示�?图像物体分割：则仅仅在一副图中生成不同物体的轮廓颜色即可，颜色自己随便填充�?
在FCN这篇论文中，我们用到的数据集即是基本的分割数据集，一共有两套分别是benchmark_RELEASE和VOC2012

图像分割的数据集一般都是采用上面说明的VOC2012挑战数据集，有人说benchmark_LELEASE为增强数据集，具体原因我不清楚，
可能是因为benchmark_LELEASE的图片都是用于分割（一�?1355张），而VOC2012仅仅部分图片适用于分割（2913张）�?"""
def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open("datapath.json").read()
    data = json.loads(js)
    return os.path.expanduser(data[name]["data_path"])


class pascalVOCLoader(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.
    
    Download:http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    Download: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

    来自VOC和SBD两个数据集的Annotations（在VOC中是RGB图像且颜色代表特定的类，在SBD中是.mat的格式）
    转换为通用的“label_mask”格式�?在此格式下，每个掩码�?�?1之间的整数值的（M，N）数组，其中0表示背景类�?   
    label masks存储在一个名为`pre_encoded`的新文件夹中�?    该文件夹作为原始Pascal VOC数据目录中`SegmentationClass`文件夹的子目录添加�?
    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(self,root,split="train_aug",is_transform=False,img_size=512,augmentations=None,img_norm=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        for split in ["train", "val", "trainval"]:
            # 分别读取VOC2012/ImageSets/Segmentation下的三个文件列表，包含训练信息，并用字典保存
            path = pjoin(self.root, "ImageSets/Segmentation", split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        self.setup_annotations()
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    def __len__(self):
        # 返回所选data splits长度，即不同训练集返回相应的训练数据个数
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded", im_name + ".png")
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        if self.img_size == ('same', 'same'):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0
        return img, lbl

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        编码mask，将颜色转换为类别标�?        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        解码mask,类别转换为颜色，方便显示
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """
        将benmark中的训练数据加入
        Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        sbd_path = get_data_path("sbd")
        target_path = pjoin(self.root, "SegmentationClass/pre_encoded")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        path = pjoin(sbd_path, "dataset/train.txt")
        sbd_train_list = tuple(open(path, "r"))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
        train_aug = self.files["train"] + sbd_train_list

        # keep unique elements (stable)
        train_aug = [
            train_aug[i] for i in sorted(np.unique(train_aug, return_index=True)[1])
        ]
        self.files["train_aug"] = train_aug
        set_diff = set(self.files["val"]) - set(train_aug)  # remove overlap
        self.files["train_aug_val"] = list(set_diff)

        pre_encoded = glob.glob(pjoin(target_path, "*.png"))
        expected = np.unique(self.files["train_aug"] + self.files["val"]).size

        if len(pre_encoded) != expected:
            print("Pre-encoding segmentation masks...")
            for ii in tqdm(sbd_train_list):
                lbl_path = pjoin(sbd_path, "dataset/cls", ii + ".mat")
                data = io.loadmat(lbl_path)
                lbl = data["GTcls"][0]["Segmentation"][0].astype(np.int32)
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(pjoin(target_path, ii + ".png"), lbl)

            for ii in tqdm(self.files["trainval"]):
                fname = ii + ".png"
                lbl_path = pjoin(self.root, "SegmentationClass", fname)
                lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(pjoin(target_path, fname), lbl)

        assert expected == 9733, "unexpected dataset sizes"


# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
# if __name__ == '__main__':
# # local_path = '/home/meetshah1995/datasets/VOCdevkit/VOC2012/'
# bs = 4
# augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
# dst = pascalVOCLoader(root=local_path, is_transform=True, augmentations=augs)
# trainloader = data.DataLoader(dst, batch_size=bs)
# for i, data in enumerate(trainloader):
# imgs, labels = data
# imgs = imgs.numpy()[:, ::-1, :, :]
# imgs = np.transpose(imgs, [0,2,3,1])
# f, axarr = plt.subplots(bs, 2)
# for j in range(bs):
# axarr[j][0].imshow(imgs[j])
# axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
# plt.show()
# a = raw_input()
# if a == 'ex':
# break
# else:
# plt.close()
