# Self-Tuning for Data-Efficient Deep Learning


## Dependencies
* python3.6
* torch == 1.3.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.4.2
* tensorboardX
* numpy
* argparse




## Quick Start
- The running commands for several datasets are shown below. Please refer to ``run.sh`` for commands for datasets with other label ratios.
```
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 15 --pretrained
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained
python src/main.py  --root ./cifar100 --batch_size 20 --logdir vis/ --gpu_id 3 --queue_size 32 --backbone efficientnet-b2 --num_labeled 10000 --expand_label --pretrained --projector_dim 1024
python src/main.py --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained --pretrained_path ./ckp-cub200/checkpoint_0099.pth.tar
python src/main.py --root ./StanfordCars --batch_size 16 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 15 --pretrained --pretrained_path ./ckp-car/checkpoint_0099.pth.tar
python src/main.py --root ./Aircraft --batch_size 16 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 15 --pretrained --pretrained_path ./ckp-air/checkpoint_0099.pth.tar


```

## Experimental Results
## Step1+Step2, using CE/CL loss fune-tune MoCo
| Dataset | Label Ratio  | no loss | CE loss | CE + CL loss | CL loss |
| -- | -- | -- | -- | -- | -- |
| CUB-200-2011 | 15%| 20.72% | 36.28% | 44.52% | 48.24% |

## calculate the acc and num of pseudo unlabeled data to be correct on CUB200 15% in first loop
## fix confidence (Con=0.95), change number of cluster (N_C)
| Methods | N_C  | N_C | N_C | N_C | N_C |N_C  | N_C |N_C |
| -- | -- | -- | -- | -- | -- | -- | -- |  -- | 
|          |all(1577)|1200|1000|800 |600 |400|200 | 100| 
| Current | 82.63%/395|82.63%/395|82.63%/395|82.63%/395|82.63%/395 | 82.63%/395|82.63%/395|82.63%/395|
| Ours  |  86.23%/332|86.78%/302|86.82%/270|86.96%/240|86.05%/185|85.99%/135|88.30%/83 |86.44%/51|

## fix number of cluster (N_C=200), change confidence (Con)
| Methods | Con  | Con | Con | Con |Con |Con  | Con |
| -- | -- | -- | -- | -- | -- | -- | -- | 
|           |0.8  | 0.9|0.95|0.96|0.97|0.98|0.98|
| Current|68.51%/790 | 75.55%/544 |82.63%/395|83.25%/343|85.42%/287|88.46%/230|90.12%/146|
| Ours   |82.12%/147 | 85.07%/114|88.30%/83|90.36%/75|91.04%/61|91.11%/41|100.00%/37|

## SSL results N_C=200, Con=0.98
## classification accuracy (%) of SSL on CUB200 
| Methods |15%|30%|50% |
| -- | -- | -- | -- | 
| Self-tuning paper |64.17|75.13|80.22|
| Self-tuning repro   |64.32|74.53|80.31|
| Ours   |66.08|76.89|80.89|

## classification accuracy (%) of SSL on StanfordCars 
| Methods |15%|30%|50% |
| -- | -- | -- | -- | 
| Self-tuning paper |72.50|83.58|88.11|
| Self-tuning repro | 75.08 |85.74|89.79|
| Ours   |78.32|88.02|91.08|

## classification accuracy (%) of SSL on Aircraft 
| Methods |15%|30%|50% |
| -- | -- | -- | -- |
| Self-tuning paper|64.11|76.03|81.22|
| Self-tuning repro | 66.92 |79.87|84.76|
| Ours  | 68.11 |82.03|86.74|



