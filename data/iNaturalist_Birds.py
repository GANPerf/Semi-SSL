import os
from typing import Optional
from .imagelist import ImageList
from ._util import *


class iNaturalist_Birds(ImageList):
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
        "train2": "image_list/train_2.txt",
        "test": "image_list/test.txt",
        "test100": "image_list/test.txt",
    }
    CLASSES = ['202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220',
               '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240',
               '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260',
               '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
               '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300',
               '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320',
               '321', '322', '323', '324', '325', '326', '327']

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

        super(iNaturalist_Birds, self).__init__(root, iNaturalist_Birds.CLASSES, data_list_file=data_list_file, **kwargs)

