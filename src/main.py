import sys
import os
proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.classifier import Classifier
from models.method import SelfTuning
from tensorboardX import SummaryWriter
from src.utils import load_network, load_data
from sklearn.preprocessing import normalize
import pandas as pd
from collections import Counter


def test_cifar(loader, model, classifier, device):
    with torch.no_grad():
        start_test = True
        iter_val = iter(loader['test'])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            feat = model.inference(inputs)
            outputs = classifier(feat)

            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def test(loader, model, classifier, device):
    with torch.no_grad():
        model.eval()
        classifier.eval()
        start_test = True
        val_len = len(loader['test0'])
        iter_val = [iter(loader['test' + str(i)]) for i in range(10)]
        for _ in range(val_len):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)
            outputs = []
            for j in range(10):
                feat = model.inference(inputs[j])
                output = classifier(feat)
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_outputs = outputs.data.float()
                all_labels = labels.data.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.data.float()), 0)
                all_labels = torch.cat((all_labels, labels.data.float()), 0)
        _, predict = torch.max(all_outputs, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])
    return accuracy

def train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=None, writer=None, model_path = None):

    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    best_acc = 0.0
    best_model = None


    #step2: Using labeled data to fine-tuning MOCOv2
    for iter_num in range(1, args.max_iter + 1):  #args.max_iter + 1   10000 is enough for convergence.
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
        #if iter_num % len_unlabeled == 0:
            #iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

        data_labeled = iter_labeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)
        path = data_labeled[2]



        ## For Labeled Data
        PGC_logit_labeled, PGC_label_labeled, feat_labeled = model(img_labeled_q, img_labeled_k, label)
        out = classifier(feat_labeled)
        classifier_loss = criterions['CrossEntropy'](out, label)
        PGC_loss_labeled = criterions['KLDiv'](PGC_logit_labeled, PGC_label_labeled)  #Contrastive loss for instances with the same labels

        #Alternative CE loss or CL loss or both. CE: using classifier_loss to fine tune MOCOv2; CL: using (pos1+pos2)/(pos1+pos2+neg) to fine tune
        total_loss = PGC_loss_labeled + classifier_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))


    
    # step3: For Unlabeled Data, Divide U data into N clusters
    # Using numpy because our gpu memory is limited T_T
    data = np.zeros((1, 2048))
    label = np.zeros(1)
    pseudo_label = np.zeros(1)
    confidence= np.zeros(1)
    arr_path = ['first']
    for i, (images, target, path) in enumerate(dataset_loaders["unlabeled_train"]):
        model.encoder_q.eval()

        images = images[0].to(device)
        #img_unlabeled_k = data_unlabeled[0][1].to(device)

        #arrange pseudo label
        _, q_f_unlabeled = model.encoder_q(images)
        logit_unlabeled = classifier(q_f_unlabeled)
        prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
        confidence_unlabeled, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)

        #change feature, pseudo label, confidence from tensor to numpy
        q_f_unlabeled = q_f_unlabeled.cpu().detach().numpy()
        predict_unlabeled = predict_unlabeled.cpu().detach().numpy()
        confidence_unlabeled = confidence_unlabeled.cpu().detach().numpy()

        #store feature, real label, preudo label and confidence -->data, label, pseudo_label, confidence
        data = np.concatenate((data, q_f_unlabeled), axis=0)
        label = np.concatenate((label, target), axis=0)
        pseudo_label = np.concatenate((pseudo_label, predict_unlabeled), axis=0)
        confidence = np.concatenate((confidence, confidence_unlabeled), axis=0)
        arr_path = np.concatenate((arr_path, path), axis=0)

    #delete the first row
    data = np.delete(data,(0), axis = 0)
    label = np.delete(label, (0), axis=0)
    pseudo_label = np.delete(pseudo_label, (0), axis=0)
    confidence = np.delete(confidence, (0), axis=0)
    arr_path = np.delete(arr_path, (0), axis=0)
    #normalize data
    data = data/ np.linalg.norm(data, axis=1).reshape(-1,1)

    #Initialize cluster label as all 0
    cluster = np.zeros(data.shape[0],dtype=np.int64)

    #check accuracy of mocov2 in retrieval, the higher, the better
    top1 = accuracy_top1(data,label)
    print("the current acc of mocov2 in retrieval after step1 and step2: {}".format(top1))

    #generate cluster label in unlabeled data. details see unlabeled_cluster.csv file
    generate_cluster(data,label,pseudo_label, cluster, confidence, arr_path)


    #select suitable unlabeled data, arrange psuedo label as part of labeled data
    select_unlabel_data(args)  #generate "select_suitable_unlabeled_data.csv" file

    select_suitable_unlabeled_data = pd.read_csv('./select_suitable_unlabeled_data.csv')
    with open('./CUB200/image_list/right_psuedo_train.txt','w') as f:
        for line in select_suitable_unlabeled_data.values:
            f.write((str(line[0][9:]))+'\t'+str(int(line[2]))+'\n')

            
    #step 4-6
    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])

    len_unlabeled = len(dataset_loaders["unlabeled_train"])   #right_psuedo_train/unlabeled_train
    iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

    len_unlabeled_psuedo = len(dataset_loaders["right_psuedo_train"])  # right_psuedo_train/unlabeled_train
    iter_unlabeled_psuedo = iter(dataset_loaders["right_psuedo_train"])

    for iter_num in range(1, args.max_iter + 1):
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
        if iter_num % len_unlabeled == 0:
            iter_unlabeled = iter(dataset_loaders["unlabeled_train"])
        if iter_num % len_unlabeled_psuedo == 0:
            iter_unlabeled_psuedo = iter(dataset_loaders["right_psuedo_train"])

        data_labeled = iter_labeled.next()
        data_unlabeled = iter_unlabeled.next()
        data_unlabeled_psuedo = iter_unlabeled_psuedo.next()

        img_labeled_q = data_labeled[0][0].to(device)
        img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)

        img_unlabeled_q = data_unlabeled[0][0].to(device)
        img_unlabeled_k = data_unlabeled[0][1].to(device)

        img_unlabeled_q_psuedo = data_unlabeled_psuedo[0][0].to(device)
        img_unlabeled_k_psuedo = data_unlabeled_psuedo[0][1].to(device)
        pseudo_label = data_unlabeled_psuedo[1].to(device)

        ## For Labeled Data
        PGC_logit_labeled, PGC_label_labeled, feat_labeled = model(img_labeled_q, img_labeled_k, label)
        out = classifier(feat_labeled)
        classifier_loss = criterions['CrossEntropy'](out, label)
        PGC_loss_labeled = criterions['KLDiv'](PGC_logit_labeled, PGC_label_labeled)  #Contrastive loss for instances with the same labels


        ## For Unlabeled Data
        q_c_unlabeled, q_f_unlabeled = model.encoder_q(img_unlabeled_q)
        logit_unlabeled = classifier(q_f_unlabeled)

        q_c_unlabeled_psuedo, q_f_unlabeled_psuedo = model.encoder_q(img_unlabeled_q_psuedo)
        logit_unlabeled_psuedo = classifier(q_f_unlabeled_psuedo)
        #classifier_unlabel_loss = criterions['CrossEntropy'](logit_unlabeled_psuedo, pseudo_label)  #pseudo_label will be from Step 3

        prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
        confidence_unlabeled, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)
        PGC_logit_unlabeled, PGC_label_unlabeled, feat_unlabeled = model(img_unlabeled_q, img_unlabeled_k, predict_unlabeled) #predict_unlabeled/pseudo_label
        PGC_loss_unlabeled = criterions['KLDiv'](PGC_logit_unlabeled, PGC_label_unlabeled)

        #prob_unlabeled_psuedo = torch.softmax(logit_unlabeled_psuedo.detach(), dim=-1)
        #confidence_unlabeled_psuedo, predict_unlabeled_psuedo = torch.max(prob_unlabeled_psuedo, dim=-1)
        PGC_logit_unlabeled_psuedo, PGC_label_unlabeled_psuedo, feat_unlabeled_psuedo = model(img_unlabeled_q_psuedo, img_unlabeled_k_psuedo, pseudo_label)  # predict_unlabeled/pseudo_label
        PGC_loss_unlabeled_psuedo = criterions['KLDiv'](PGC_logit_unlabeled_psuedo, PGC_label_unlabeled_psuedo)

        total_loss = classifier_loss + PGC_loss_labeled + PGC_loss_unlabeled + PGC_loss_unlabeled_psuedo #+ classifier_unlabel_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()


        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))

        ## Show Loss in TensorBoard
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/classifier_loss', classifier_unlabel_loss, iter_num)
        #writer.add_scalar('loss/PGC_loss_labeled', PGC_loss_labeled, iter_num)
        #writer.add_scalar('loss/PGC_loss_unlabeled', PGC_loss_unlabeled, iter_num)
        writer.add_scalar('loss/total_loss', total_loss, iter_num)

        if iter_num % args.test_interval == 1 or iter_num == 500:
            model.eval()
            classifier.eval()
            if 'cifar100' in args.root:
                test_acc = test_cifar(dataset_loaders, model, classifier, device=device)
            else:
                test_acc = test(dataset_loaders, model, classifier, device=device)
            print("iter_num: {}; test_acc: {}".format(iter_num, test_acc))
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = {'model': model.state_dict(),
                              'classifier': classifier.state_dict(),
                              'step': iter_num
                              }
    print("best acc: %.4f" % (best_acc))
    torch.save(best_model, model_path)
    print("The best model has been saved in ", model_path)

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
    parser.add_argument('--num_of_cluster', type=float, default=1200)
    parser.add_argument('--confidence', type=float, default=0.95)
    ## Only for Cifar100
    parser.add_argument("--expand_label", action="store_true", help="expand label to fit eval steps")
    parser.add_argument('--num_labeled', type=int, default=0, help='number of labeled data')
    configs = parser.parse_args()
    return configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    args = read_config()
    set_seed(args.seed)

    # Prepare data
    if 'cifar100' in args.root:
        args.class_num = 100
    elif 'CUB200' in args.root:
        args.class_num = 200
    elif 'StanfordCars' in args.root:
        args.class_num = 196
    elif 'Aircraft' in args.root:
        args.class_num = 100

    dataset_loaders = load_data(args)
    print("class_num: ", args.class_num)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    if 'cifar100' in args.root:
        model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.num_labeled))
    else:
        model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.label_ratio))
    logdir = os.path.join(args.logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    model_path = os.path.join(logdir, "%s_best.pkl" % (model_name))

    # Step1: Initialize model, using pretrained MOCO v2 in pretrained_path
    network, feature_dim = load_network(args.backbone)
    model = SelfTuning(network=network, backbone=args.backbone, queue_size=args.queue_size, projector_dim=args.projector_dim, feature_dim=feature_dim,
                       class_num=args.class_num, momentum=args.momentum, pretrained=args.pretrained, pretrained_path=args.pretrained_path).to(device)
    classifier = Classifier(feature_dim, args.class_num).to(device)
    print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))

    ## Define Optimizer
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr= args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # Train model
    train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=device, writer=writer, model_path=model_path)

def accuracy_top1(data, label):
    topN1 = []
    for j in range(data.shape[0]):
        query_feat = data[j, :].reshape(1, 2048)  # 1*1000
        query_label = label[j]

        dict = np.delete(data, (j), axis=0)  # 5793*1000
        sim_label = np.delete(label, (j), axis=0)  # 5793

        similarity = query_feat.dot(dict.T)  # 1*5793

        # https://blog.csdn.net/Zhang_314/article/details/81255329
        table = np.zeros((similarity.shape[1], 2))
        table[:, 0] = similarity.squeeze()
        table[:, 1] = sim_label

        index = np.argsort(table[:, 0])[::-1]

        T = table[index]
        # https://wrong.wang/blog/20190223-reid%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84cmc%E5%92%8Cmap/
        # to calculate top-1
        if T[0, 1] == query_label:
            topN1.append(1)
        else:
            topN1.append(0)

    top1 = np.mean(topN1)
    return top1

def generate_cluster(data, label, pseudo_label, cluster,confidence,arr_path):
    j = 1
    i = 0
    while i < data.shape[0]:
        while cluster[i] != 0:
            if i >= data.shape[0] - 1:
                break
            i = i + 1

        cluster[i] = j
        query_feat = data[i, :].reshape(1, 2048)
        list1 = [i]
        id = i
        while True:
            # find similarity index
            similarity = query_feat.dot(data.T)
            similarity = np.squeeze(similarity)
            similarity[id] = 0.0
            index = np.argsort(-similarity)
            id = index[0]
            if cluster[id] == 0:
                list1.append(id)
                cluster[id] = j
                query_feat = data[id, :]
            else:
                for k in range(len(list1)):
                    data[list1[k], :] = np.zeros(2048)
                break
        i = i + 1
        j = j + 1

    dataframe = pd.DataFrame({'image':arr_path, 'real label': label, 'cluster label': cluster,'pseudo_label': pseudo_label,'confidence_unlabeled':confidence})
    dataframe.to_csv("unlabeled_cluster.csv", index=False)

def select_unlabel_data(args):
    unlabel_cluster = pd.read_csv("./unlabeled_cluster.csv")
    list1 = unlabel_cluster.values.tolist()
    arr = np.array(list1)

    temp = arr[:, 4]
    number = temp.tolist()
    list_confidence = list(map(float, number))
    array_confidence = np.array(list_confidence)
    row_index_confidence = np.where(array_confidence >= args.confidence)
    high_confidence_arr = arr[row_index_confidence,:][0]
    select_total_unlabel_data = arr[0, :].reshape(1, -1)


    j = 1
    temp = arr[:, 2]
    number = temp.tolist()
    # print(max(list(map(int, number))))

    while j <= args.num_of_cluster and j <= max(list(map(int, number))):
        k = str(j)
        row_index = np.where(arr[:, 2] == k)
        sub_arr = arr[row_index, :][0]

        most_psuedo_label = Counter(sub_arr[:, 3]).most_common(1)[0][0]

        for i in range(sub_arr.shape[0]):
            if sub_arr[i, 3] == most_psuedo_label and float(sub_arr[i, 4]) >= args.confidence:
                select_total_unlabel_data = np.concatenate((select_total_unlabel_data, sub_arr[i, :].reshape(1, -1)),
                                                           axis=0)

        j = j + 1

    # delete the first row
    select_total_unlabel_data = np.delete(select_total_unlabel_data, (0), axis=0)


    hit_num = (select_total_unlabel_data[:, 3] == select_total_unlabel_data[:, 1]).sum()
    sample_num = select_total_unlabel_data.shape[0]
    print("Current num:{}; current acc of psuedo label in our method: {}".format(hit_num, hit_num / float(sample_num)))

    hit_num = (high_confidence_arr [:, 3] == high_confidence_arr [:, 1]).sum()
    sample_num = high_confidence_arr.shape[0]
    print("Current num:{}; current acc of psuedo label in current paper: {}".format(hit_num, hit_num / float(sample_num)))

    dataframe = pd.DataFrame(
        {'image': select_total_unlabel_data[:, 0], 'real label': select_total_unlabel_data[:, 1],
         'pseudo_label': select_total_unlabel_data[:, 3],
         'confidence_unlabeled': select_total_unlabel_data[:, 4], 'cluster label': select_total_unlabel_data[:, 2], })
    dataframe.to_csv("select_suitable_unlabeled_data.csv", index=False)

if __name__ == '__main__':
    main()

