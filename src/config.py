import argparse


def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--label_ratio', type=int, default=15)
    parser.add_argument('--logdir', type=str, default='../vis/')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--seed', type=int, default='666666')
    parser.add_argument('--workers', type=int, default='4')
    parser.add_argument('--lr_ratio', type=float, default='10')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--queue_size', type=int, default=32, help='queue size for each class')
    parser.add_argument('--momentum', type=float, default=0.999, help='the momentum hyperparameter for moving average')
    parser.add_argument('--projector_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--max_iter', type=float, default=27005)
    parser.add_argument('--test_interval', type=float, default=3000)
    parser.add_argument("--pretrained", action="store_true", help="use the pre-trained model")
    parser.add_argument("--pretrained_path", type=str, default='~/.torch/models/moco_v2_800ep_pretrain.pth.tar')
    parser.add_argument('--num_of_cluster', type=float, default=200)
    parser.add_argument('--confidence', type=float, default=0.98)
    ## Only for Cifar100
    parser.add_argument("--expand_label", action="store_true", help="expand label to fit eval steps")
    parser.add_argument('--num_labeled', type=int, default=0, help='number of labeled data')
    configs = parser.parse_args()
    return configs
