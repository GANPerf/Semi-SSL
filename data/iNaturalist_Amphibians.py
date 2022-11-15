import os
from typing import Optional
from .imagelist import ImageList
from ._util import *


class iNaturalist_Amphibians(ImageList):
    """`The Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        label_ratio (int): The label rates to randomly sample ``training`` images for each category.
            Choices include 100, 50, 30, 15. Default: 100.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
            test/
            image_list/
                train_100.txt
                train_50.txt
                train_30.txt
                train_15.txt
                test.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d95c188cc49c404aba70/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/d5ab63c391a949509db0/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/04e6fd5222a84d0a8ff5/?dl=1"),
    ]
    image_list = {
        "train": "image_list/train_100.txt",
        "train100": "image_list/train_100.txt",
        "train5": "image_list/train_5.txt",
        "test": "image_list/test.txt",
        "test100": "image_list/test.txt",
    }
    CLASSES = ['153', '154', '155', '156', '157', '158', '159', '160', '161', '162']

    def __init__(self, root: str, split: str, label_ratio: Optional[int] =100, download: Optional[bool] = False, **kwargs):

        if split == 'train':
            list_name = 'train' + str(label_ratio)
            assert list_name in self.image_list
            data_list_file = os.path.join(root, self.image_list[list_name])
        elif split == 'unlabeled_train':
            data_list_file = os.path.join(root, "image_list/unlabeled_" + str(label_ratio) + ".txt")
            # if not os.path.exists(data_list_file):
            train_list_name = 'train' + str(label_ratio)
            full_list_name = 'train'
            assert train_list_name in self.image_list
            assert full_list_name in self.image_list
            train_list_file = os.path.join(root, self.image_list[train_list_name])
            full_list_file = os.path.join(root, self.image_list[full_list_name])
            train_list = read_list_from_file(train_list_file)
            full_list = read_list_from_file(full_list_file)
            unlabel_list = list(set(full_list) - set(train_list))
            save_list_to_file(data_list_file, unlabel_list)
        else:
            data_list_file = os.path.join(root, self.image_list['test'])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(iNaturalist_Amphibians, self).__init__(root, iNaturalist_Amphibians.CLASSES, data_list_file=data_list_file, **kwargs)

