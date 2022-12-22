import math
import os

import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision import transforms

from data.randaugment import RandAugmentMC

cub200_mean =(0.485, 0.456, 0.406)
cub200_std =(0.229, 0.224, 0.225)

img_size=224
crop_size=224
resize_size = 256

class ResizeImage(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            output size will be (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url ='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'# 'https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/' #''http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'


    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, indexs, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.classes=None
        self.targets = None
        self.indexs=indexs
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        image_classes=pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                       sep=' ', names=['cls_id', 'cls_name'])

        self.classes = image_classes['cls_name'].tolist()#image_classes.merge(classes, on='cls_id')
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
            if len(self.indexs) >0:
                self.data=self.data.iloc[self.indexs]
                self.data.target = np.array(self.data.target.iloc[self.indexs])
        else:
            self.data = self.data[self.data.is_training_img == 0]
            if len(self.indexs) >0:
                self.data=self.data[self.indexs]
                self.data.target = np.array(self.data.target[self.indexs])
        self.targets = self.data.target.tolist()
        self.targets=[x-1 for x in self.targets ]


    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target-1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    #assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            ResizeImage(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            ResizeImage(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
def get_cub200(args, root):

    transform_labeled = transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=crop_size,
                              padding=int(crop_size*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cub200_mean, std=cub200_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cub200_mean, std=cub200_std)])

    base_dataset = Cub2011(args.root, indexs=[], train=True, transform=None, loader=default_loader, download=args.download)

    args.num_labeled=len(base_dataset.targets)
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = Cub2011(
        args.root, train_labeled_idxs.tolist(), train=True,
        transform=transform_labeled,download=False)

    train_unlabeled_dataset = Cub2011(
        args.root, train_unlabeled_idxs.tolist(), train=True,
        transform=TransformFixMatch(mean=cub200_mean, std=cub200_std),download=args.download)

    test_dataset = Cub2011(
        args.root, [],train=False, transform=transform_val, download=args.download)  # by default, the index is none to retrieve test set.

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


DATASET_GETTERS = {'cub200_2011': get_cub200,
                   }