import torch
import torchvision
from torchvision.transforms import transforms

from data.randaugment import RandAugmentMC
from models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101
from models.efficientnet import EfficientNetFc

from data.tranforms import TransformTrain
from data.tranforms import TransformTest
from data.cub200_2011 import get_cub200
from data.cifar100 import get_cifar100
from torch.utils.data import DataLoader, RandomSampler
import os

imagenet_mean=(0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)

cub200_mean =(0.485, 0.456, 0.406)
cub200_std =(0.229, 0.224, 0.225)

img_size=224
crop_size=224
resize_size = 256

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
transform_labeled = transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=crop_size,
                              padding=int(crop_size*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cub200_mean, std=cub200_std)])

transform_val = transforms.Compose([
    ResizeImage(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=cub200_mean, std=cub200_std)])
def load_data(args):
    batch_size_dict = {"train": args.batch_size, "unlabeled_train": args.batch_size,"test": 100}    #"right_psuedo_train": args.batch_size

    if 'cifar100' in args.root:
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, args.root)
        labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=RandomSampler(labeled_dataset),
            batch_size=batch_size_dict["train"],
            num_workers=args.num_works,
            drop_last=True)

        unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=RandomSampler(unlabeled_dataset),
            batch_size=batch_size_dict["unlabeled_train"],
            num_workers=args.num_works,
            drop_last=True)

        ## We didn't apply tencrop test since other SSL baselines neither
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_dict["test"],
            shuffle=False,
            num_workers=args.num_workers)

        dataset_loaders = {"train": labeled_trainloader,
                           "unlabeled_train": unlabeled_trainloader,
                           "test": test_loader}

    elif args.fixmatch==True:
        if args.dataset=='cub_200_2011':
            labeled_dataset, unlabeled_dataset, test_dataset = get_cub200(args)
            return labeled_dataset, unlabeled_dataset, test_dataset
        elif args.dataset in ['cub200','stanfordcars','aircrafts']:
            proportions = [args.label_ratio, 1-args.label_ratio]
            unlabeled_set = torchvision.datasets.ImageFolder(
                os.path.join(args.root, 'train'),
                transform=TransformFixMatch(mean=cub200_mean, std=cub200_std))  # StandfordCars(data_dir,train=train,transform=transform)
            lengths = [int(p * len(unlabeled_set)) for p in proportions]
            lengths[-1] = len(unlabeled_set) - sum(lengths[:-1])
            label_set = torchvision.datasets.ImageFolder(
                os.path.join(args.root, 'train'),
                transform=transform_labeled)
            labeled_set,_=torch.utils.data.random_split(label_set,  lengths)
            test_set = torchvision.datasets.ImageFolder(
                os.path.join(args.root, 'test'),
                transform=transform_val)  # StandfordCars(data_dir,train=train,transform=transform)
            return labeled_set,unlabeled_set,test_set

    else:
        transform_train = TransformTrain()
        transform_test = TransformTest(mean=imagenet_mean, std=imagenet_std)
        dataset = data.__dict__[os.path.basename(args.root)]

        datasets = {"train": dataset(root=args.root, split='train', label_ratio=args.label_ratio, download=True, transform=transform_train),
                    "unlabeled_train": dataset(root=args.root, split='unlabeled_train', label_ratio=args.label_ratio, download=True, transform=transform_train)}
                    #"right_psuedo_train": dataset(root=args.root, split='right_psuedo_train', label_ratio=args.label_ratio, download=True, transform=transform_train)
        test_dataset = {
            'test' + str(i): dataset(root=args.root, split='test', label_ratio=100, download=True, transform=transform_test["test" + str(i)]) for i in range(10)
        }
        datasets.update(test_dataset)

        dataset_loaders = {x: DataLoader(datasets[x], batch_size=batch_size_dict[x], shuffle=True, num_workers=args.num_workers)
                           for x in ['train', 'unlabeled_train']}
        dataset_loaders.update({'test' + str(i): DataLoader(datasets["test" + str(i)], batch_size=4, shuffle=False, num_workers=args.num_workers)
                                for i in range(10)})

    return dataset_loaders



def load_network(backbone):
    if 'resnet' in backbone:
        if backbone == 'resnet18':
            network = resnet18
            feature_dim = 512
        elif backbone == 'resnet34':
            network = resnet34
            feature_dim = 512
        elif backbone == 'resnet50':
            network = resnet50
            feature_dim = 2048
        elif backbone == 'resnet101':
            network = resnet101
            feature_dim = 2048
        elif backbone == 'resnet152':
            network = resnet152
            feature_dim = 2048
    elif 'efficientnet' in backbone:
        network = EfficientNetFc
        print(backbone)
        if backbone == 'efficientnet-b0':
            feature_dim = 1280
        elif backbone == 'efficientnet-b1':
            feature_dim = 1280
        elif backbone == 'efficientnet-b2':
            feature_dim = 1408
        elif backbone == 'efficientnet-b3':
            feature_dim = 1536
        elif backbone == 'efficientnet-b4':
            feature_dim = 1792
        elif backbone == 'efficientnet-b5':
            feature_dim = 2048
        elif backbone == 'efficientnet-b6':
            feature_dim = 2304
    else:
        network = resnet50
        feature_dim = 2048

    return network, feature_dim