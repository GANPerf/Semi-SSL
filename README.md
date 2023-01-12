# A new Curriculum Pseudo-Labeling Strategy: Improving Semi-supervised Learning via Consistent and Cluster Pseudo Labeling


## Dependencies
* python3.6
* torch == 1.3.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.4.2
* tensorboardX
* numpy
* argparse
* apex git clone https://github.com/NVIDIA/apex
* cd apex
* python setup.py install

## Hardware minimum requirements
when batch size =64 on cub/stanfordcars/aircrafts, it requires minumum 50G GPU memory


## Quick Start
- The running commands for several datasets are shown below. Please refer to ``run.sh`` for commands for datasets with other label ratios.
```
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 15 --pretrained --confidence 0.98
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained --confidence 0.98
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained --confidence 0.98
python src/main.py  --root ./cifar100 --batch_size 20 --logdir vis/ --gpu_id 3 --queue_size 32 --backbone efficientnet-b2 --num_labeled 10000 --expand_label --pretrained --projector_dim 1024
python src/main.py --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained --pretrained_path ./ckp-cub200/checkpoint_0099.pth.tar
python src/main.py --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 15 --pretrained --pretrained_path ./ckp-car/checkpoint_0099.pth.tar
python src/main.py --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 15 --pretrained --pretrained_path ./ckp-air/checkpoint_0099.pth.tar


```

## Experimental Results
## Step1+Step2, using CE/CL loss fune-tune MoCo (%)
| Dataset | Label Ratio  | no loss | CE loss | CE + CL loss | CL loss |
| -- | -- | -- | -- | -- | -- |
| CUB-200-2011 | 15| 20.72 | 36.28 | 44.52 | 48.24 |
| StanfordCars | 15| 10.66 | 41.49 | 52.75 | 54.68 |
| Aircraft | 15| 24.14 | 41.27 | 47.62 | 50.46 |

## calculate the acc and num of pseudo unlabeled data to be correct on CUB200 15% in first loop
## fix confidence (Con=0.95), change number of cluster (N_C)
| Methods | N_C  | N_C | N_C | N_C | N_C |N_C  | N_C |N_C |
| -- | -- | -- | -- | -- | -- | -- | -- |  -- | 
|          |all(1577)|1200|1000|800 |600 |400|200 | 100| 
| Classifier_CE | 82.63%/395|82.63%/395|82.63%/395|82.63%/395|82.63%/395 | 82.63%/395|82.63%/395|82.63%/395|
| Ours  |  86.23%/332|86.78%/302|86.82%/270|86.96%/240|86.05%/185|85.99%/135|88.30%/83 |86.44%/51|

## fix number of cluster (N_C=200), change confidence (Con)
| Methods | Con  | Con | Con | Con |Con |Con  | Con |
| -- | -- | -- | -- | -- | -- | -- | -- | 
|           |0.8  | 0.9|0.95|0.96|0.97|0.98|0.99|
| Classifier_CE|68.51%/790 | 75.55%/544 |82.63%/395|83.25%/343|85.42%/287|88.46%/230|90.12%/146|
| Ours   |82.12%/147 | 85.07%/114|88.30%/83|90.36%/75|91.04%/61|91.11%/41|100.00%/37|

## SSL results N_C=200, Con=0.98
## classification accuracy (%) of SSL on CUB200 
| Methods |15%|30%|50% |
| -- | -- | -- | -- | 
| Self-tuning paper |64.17±0.47|75.13±0.35|80.22+0.36|
| Self-tuning repro   |64.32+0.42|74.53±0.27|80.31±0.32|
| Ours   |66.08±0.26|76.89±0.43|80.89±0.28|

## classification accuracy (%) of SSL on StanfordCars 
| Methods |15%|30%|50% |
| -- | -- | -- | -- | 
| Self-tuning paper |72.50±0.45|83.58±0.28|88.11±0.29|
| Self-tuning repro | 75.08±0.34 |85.74±0.36|89.79±0.26|
| Ours   |78.32±0.26|88.02±0.44|91.08±0.18|

## classification accuracy (%) of SSL on Aircraft 
| Methods |15%|30%|50% |
| -- | -- | -- | -- |
| Self-tuning paper|64.11±0.32|76.03±0.25|81.22±0.29|
| Self-tuning repro | 66.92±0.18 |79.87±0.21|84.76±0.27|
| Ours  | 68.11±0.17 |82.03±0.22|86.74±0.30|

## classification accuracy (%) of SSL on Cifar100 (ResNet50)
| Methods\num_labeled |400|2500|10000 |
| -- | -- | -- | -- |
| Self-tuning paper|-|-|-|
| Self-tuning repro | 41.97 |68.78|79.09|
| Ours  |  |||

## Comparison of Pseudo Label Acc on Unlabeled Data on StanfordCars 15%
| Methods\epoch |20|40| 60| 80|100 |Semi-SL results |
| -- | -- | -- | -- |-- | -- |-- |
| Self-tuning repro|53.86|61.76|62.47|62.29|62.68|75.08|
| Ours (first loop) | 53.52 |61.61|61.98|62.43|62.65|74.57|
| Ours (second loop) |59.85 |65.50|66.15|66.75|66.27|77.28|
| Ours (third loop) | 61.85 |66.77|67.24|66.90|67.15|78.11|
| Ours (fourth loop) | 58.73 |67.56|67.16|67.63|66.98|77.60|

## Question 4: Is MOCOv2 vital in our method?
step 1 using ResNet50(pretrained=True) instead of MOCOv2

Step1+Step2, using CE/CL loss fune-tune ResNet50(pretrained=True) compared to MOCOv2
| StanfordCars (15%)   | no loss | CE loss | CE + CL loss | CL loss | Semi-SL results |
| -- | -- | -- | -- | -- | -- |
| self-tuning repro | - | - | - | - | 75.08 |
| Ours(MOCOv2)| 10.66 | 41.49 | 52.75 | 54.68| 78.32 |
| Ours(ResNet50)| 4.91 | 39.87 | 53.39 | 53.58 | 77.89 |

| Aircraft (15%)   | no loss | CE loss | CE + CL loss | CL loss | Semi-SL results |
| -- | -- | -- | -- | -- | -- |
| self-tuning repro | - | - | - | - | 66.92|
| Ours(MOCOv2)| 24.14 | 41.27 | 47.62 |50.46 | 68.11 |
| Ours(ResNet50)| 5.16 | 42.44 | 52.03 | 52.14 | 68.68 |

Answer: MOCOv2 maybe not necessary, we can use resnet50(pretrained=True) to replace MOCOv2 in our step1 and step 2

## Question 2: the internal and external double loop improve performance? 
(directly sort by confidence, and move the unlabeled set to the labeled set)

(Nothing to do with step1 and step2)

| StanfordCars (15%) | Con  | Con | Con | Con |Con |Con |Con |
| -- | -- | -- | -- | -- | -- | -- | -- | 
|           |0.80|0.90|0.95|0.96|0.97|0.98|0.99|
| Self-tuning repro|75.08|75.08|75.08|75.08|75.08|75.08|75.08|
| Ours_no_circle|71.62|73.32|75.90|75.31|76.02|75.44|75.45|
| Ours|77.63|78.09|78.72|78.31|78.93|78.32|79.21|

Answer: Yes, the internal and external double loop play a vital role in improving performance.


## Fixmatch(reproduction+ backbone=Resnet50)| classification accuracy (%) of SSL on CUB200 /StandfordCar/AirCrafts 
```
python train_fixmatch.py --dataset stanfordcars --amp --fixmatch 1 --download 0 --root /root/Projects/Semi-SSL/StanfordCars/StanfordCars  --arch resnet50 --batch-size 64 --lr 0.03 --seed 5 --out results/stanforcars@1500.5 --label_ratio .15
python -m torch.distributed.launch --nproc_per_node 2 ./train_fixmatch.py --dataset stanfordcars --amp --fixmatch 1 --download 0 --root /data/yangyang/StanfordCars --arch resnet50 --batch-size 32 --lr 0.03 --seed 5 --out results/stanforcars@1500.5 --label_ratio .15
python train_fixmatch.py  --dataset aircrafts --amp --fixmatch 1 --download 0 --root /root/Projects/Semi-SSL/Aircraft/Aircraft  --arch resnet50 --batch-size 64 --lr 0.03 --seed 5 --out results/aircrafts@1500.5 --label_ratio .15
python -m torch.distributed.launch --nproc_per_node 2 ./train_fixmatch.py --dataset our_cifar100 --amp --fixmatch 1 --download 0 --root /data/yangyang/our_cifar_100 --arch resnet50 --batch-size 32 --lr 0.03 --seed 5 --out results/cifar100@500.01 --label_ratio .01
```

| Dataset/ratio of dataset usage (labeled) |15%|30%|50% |
| -- | -- | -- | -- | 
| CUB200 | 41.30±0.28|62.56±0.16 |68.23±0.35 |
| StandfordCar |66.37±0.22 | 82.15±0.14| 85.75±0.20|
| Aircrafts | 60.19±0.17| 72.79±0.21|78.79±0.16 |

## Fixmatch(our algorithm+ backbone=Resnet50)| classification accuracy (%) of SSL on CUB200 /StandfordCar/AirCrafts 
```
python train_fixmatch.py --dataset stanfordcars --amp --fixmatch 1 --download 0 --root /root/Projects/Semi-SSL/StanfordCars/StanfordCars  --arch resnet50 --batch-size 64 --lr 0.03 --seed 5 --out results/stanforcars@1500.5 --label_ratio .15
python -m torch.distributed.launch --nproc_per_node 2 ./train_fixmatch.py --dataset stanfordcars --amp --fixmatch 1 --download 0 --root /root/Projects/Semi-SSL/StanfordCars/StanfordCars  --arch resnet50 --batch-size 64 --lr 0.03 --seed 5 --out results/stanforcars@1500.5 --label_ratio .15 --iter-train-step1 3 --modelCE_iter 20
python train_fixmatch.py  --dataset stanfordcars --amp --fixmatch 1 --download 0 --root /root/Projects/Semi-SSL/StanfordCars/StanfordCars  --arch resnet50 --batch-size 64 --lr 0.03 --seed 5 --out results/stanforcars@1500.5 --label_ratio .15 --iter-train-step1 3 --modelCE_iter 20
```

|        Dataset/ratio of dataset usage (labeled)              |            |  15%  |            |  |       | 30%   |           | |    | 50%   |            |
|----------------------|------------|------------|------------|---|------------|------------|------------|---| ------------|------------|------------|
|       |    c1   |    c2    |   c3    | |   c1   |    c2    |   c3    | |c1   |    c2    |   c3     |
| CUB200 | | | |||||||||
| StandfordCar |66.51|68.61|70.03 ||82.52|82.85|83.41||73.87(epoch=6)| ||
| AirCrafts |59.77|59.89|62.02| ||| ||80.02|80.08 ||

Note: c_n=  number of unlabeled data cycling(clustering) times.

## Classifiaction accuracy of our method and fixmatch on cifar100 with a randomly initialized Resnet-50 network (pretrained=False)

| Method/ratio of dataset usage (labeled) |1%|5%|10% |15%|20%|
| -- | -- | -- | -- | -- |-- |
| Fixmatch |screen2 | | |screen1||
| Ours | || |||
