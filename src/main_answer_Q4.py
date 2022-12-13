import sys
import os

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.classifier import Classifier
from models.method import MOCOTuning
from models.method import SelfTuning
from tensorboardX import SummaryWriter
from src.utils import load_network, load_data
import pandas as pd
from collections import Counter
from src.config import read_config

proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)


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


def is_pick_unlabeled_data(df, confidence_unlabeled):
    df1 = df.groupby(['cluster_label', 'pseudo_label']).filter(lambda x: len(x) > 1)
    if len(df1[df1['confidence_unlabeled'] > confidence_unlabeled]) >= 1:
        return True
    return False


def step3(args, classifier, dataset_loaders, device, model, model_ce):  # first mode for cycle, second for psuedo
    print('step3 starts')
    data = np.zeros((1, 2048))
    label = np.zeros(1)
    pseudo_label = np.zeros(1)
    confidence = np.zeros(1)
    arr_path = ['first']

    for i, (images, target, path) in enumerate(dataset_loaders["unlabeled_train"]):
        model.encoder_q.eval()
        classifier.eval()

        images = images[0].to(device)
        # img_unlabeled_k = data_unlabeled[0][1].to(device)

        # arrange pseudo label
        _, q_f_unlabeled = model.encoder_q(images)  # feat for retrieval using MOCOv2
        _, unlabeled_feat = model_ce(images)  # feat for arrange psuedo label using Resnet50(pretrained=True)
        logit_unlabeled = classifier(unlabeled_feat)
        prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
        confidence_unlabeled, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)

        # change feature, pseudo label, confidence from tensor to numpy
        q_f_unlabeled = q_f_unlabeled.cpu().detach().numpy()
        predict_unlabeled = predict_unlabeled.cpu().detach().numpy()
        confidence_unlabeled = confidence_unlabeled.cpu().detach().numpy()

        # store feature, real label, preudo label and confidence -->data, label, pseudo_label, confidence
        data = np.concatenate((data, q_f_unlabeled), axis=0)
        label = np.concatenate((label, target), axis=0)
        pseudo_label = np.concatenate((pseudo_label, predict_unlabeled), axis=0)
        confidence = np.concatenate((confidence, confidence_unlabeled), axis=0)
        arr_path = np.concatenate((arr_path, path), axis=0)
    # delete the first row
    data = np.delete(data, (0), axis=0)
    label = np.delete(label, (0), axis=0)
    pseudo_label = np.delete(pseudo_label, (0), axis=0)
    confidence = np.delete(confidence, (0), axis=0)
    arr_path = np.delete(arr_path, (0), axis=0)
    # normalize data
    data = data / np.linalg.norm(data, axis=1).reshape(-1, 1)
    # Initialize cluster label as all 0
    cluster = np.zeros(data.shape[0], dtype=np.int64)
    # check accuracy of mocov2 in retrieval, the higher, the better
    top1 = accuracy_top1(data, label)
    print("the current acc of mocov2 in retrieval after step1 and step2: {}".format(top1))
    # generate cluster label in unlabeled data. details see unlabeled_cluster.csv file
    df_unlabeled_cluster = generate_cluster(args, data, label, pseudo_label, cluster, confidence, arr_path)
    # select suitable unlabeled data, arrange psuedo label as part of labeled data
    df_select_unlabel_data = select_unlabel_data(args)  # generate "select_suitable_unlabeled_data.csv" file
    select_suitable_unlabeled_data = pd.read_csv(
        'select_suitable_unlabeled_data' + str(args.class_num) + str(args.label_ratio) + '.csv')
    # with open('./StanfordCars/image_list/right_psuedo_train.txt', 'w') as f:
    # for line in select_suitable_unlabeled_data.values:
    # f.write((str(line[0][9:])) + '\t' + str(int(line[2])) + '\n')
    return df_unlabeled_cluster, df_select_unlabel_data


def step2(criterions, dataset_loaders, device, iter_labeled, len_labeled, model, optimizer,
          scheduler, classifier):
    # step2: Using labeled data to fine-tuning MOCOv2
    print('step2 starts')

    for iter_num in range(1, 10000 + 1):  # args.max_iter + 1   10000 is enough for convergence.
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
        # if iter_num % len_unlabeled == 0:
        # iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

        data_labeled = iter_labeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)
        #path = data_labeled[2]

        ## For Labeled Data
        PGC_logit_labeled, PGC_label_labeled, feat_labeled = model(img_labeled_q, img_labeled_k, label)

        PGC_loss_labeled = criterions['KLDiv'](PGC_logit_labeled,
                                               PGC_label_labeled)  # Contrastive loss for instances with the same labels

        out = classifier(feat_labeled)
        classifier_loss = criterions['CrossEntropy'](out, label)

        # CL: using (pos1+pos2)/(pos1+pos2+neg) to fine tune
        total_loss = PGC_loss_labeled + classifier_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            print("step2: iter_num: {}".format(iter_num))


def step4(classifier, criterions, dataset_loaders, device, model, optimizer, scheduler):
    # step4: Using labeled data (CE loss) to fine-tuning MOCOv2
    print('step4 starts')
    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])
    for iter_num in range(1, 12000 + 1):  # args.max_iter + 1
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()

        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])

        data_labeled = iter_labeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        # img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)

        ## For Labeled Data
        _, feat_labeled = model(img_labeled_q)
        out = classifier(feat_labeled)
        classifier_loss = criterions['CrossEntropy'](out, label)

        total_loss = classifier_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {0:2d}; current acc: {1:8.2f}".format(iter_num, hit_num / float(sample_num)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def generate_cluster(args, data, label, pseudo_label, cluster, confidence, arr_path):
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

    dataframe = pd.DataFrame(
        {'image': arr_path, 'real_label': label, 'cluster_label': cluster, 'pseudo_label': pseudo_label,
         'confidence_unlabeled': confidence})
    dataframe.to_csv('unlabeled_cluster' + str(args.class_num) + str(args.label_ratio) + '.csv', index=False)
    return dataframe


def select_unlabel_data(args):
    unlabel_cluster = pd.read_csv('unlabeled_cluster' + str(args.class_num) + str(args.label_ratio) + '.csv')
    list1 = unlabel_cluster.values.tolist()
    arr = np.array(list1)

    temp = arr[:, 4]
    number = temp.tolist()
    list_confidence = list(map(float, number))
    array_confidence = np.array(list_confidence)
    row_index_confidence = np.where(array_confidence >= args.confidence)
    high_confidence_arr = arr[row_index_confidence, :][0]
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

    hit_num = (high_confidence_arr[:, 3] == high_confidence_arr[:, 1]).sum()
    sample_num = high_confidence_arr.shape[0]
    print(
        "Current num:{}; current acc of psuedo label in current paper: {}".format(hit_num, hit_num / float(sample_num)))

    dataframe = pd.DataFrame(
        {'image': select_total_unlabel_data[:, 0], 'real label': select_total_unlabel_data[:, 1],
         'pseudo_label': select_total_unlabel_data[:, 3],
         'confidence_unlabeled': select_total_unlabel_data[:, 4], 'cluster label': select_total_unlabel_data[:, 2], })
    dataframe.to_csv('select_suitable_unlabeled_data' + str(args.class_num) + str(args.label_ratio) + '.csv',
                     index=False)
    return dataframe


def train(args, model, model_ce, model_moco, classifier_step2, classifier, classifier_ce, dataset_loaders, optimizer_ce,
          optimizer_moco,scheduler_ce, scheduler_moco, device=None, writer=None, model_path=None):
    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])
    is_loop = True
    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    best_acc = 0.0
    best_model = None
    step2(criterions, dataset_loaders, device, iter_labeled, len_labeled, model_moco, optimizer_moco,
          scheduler_moco, classifier_step2)
    step4(classifier_ce, criterions, dataset_loaders, device, model_ce, optimizer_ce,
          scheduler_ce)
    # step3: For Unlabeled Data, Divide U data into N clusters
    # Using numpy because our gpu memory is limited T_T

    while is_loop:
        df_unlabeled_cluster, df_select_unlabel_data = step3(args, classifier_ce, dataset_loaders, device, model_moco,
                                                             model_ce)
        if not (is_pick_unlabeled_data(df_unlabeled_cluster, args.confidence)):
            break

        # merge selected_unlabel_data to dataset_loaders["train"]
        dataset_loaders['train'].dataset.samples.extend(list(
            zip(df_select_unlabel_data['image'], df_select_unlabel_data['pseudo_label'].astype(float).astype(int))))

        dataset_loaders["unlabeled_train"].dataset.samples = list(
            filter(lambda x: x[0] not in list(df_select_unlabel_data.loc[:, 'image'].values),
                   dataset_loaders[
                       'unlabeled_train'].dataset.samples))  # remove df_select_unlabel_data from unlabeled_train set

        # step 5-6
        print('step4-6 starts')
        len_labeled = len(dataset_loaders["train"])
        iter_labeled = iter(dataset_loaders["train"])

        len_unlabeled = len(dataset_loaders["unlabeled_train"])  # right_psuedo_train/unlabeled_train
        iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

        # reset lr, optimizer and scheduler5
        optimizer = optim.SGD([
            {'params': model.parameters()},
            {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},
        ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        milestones = [6000, 12000, 18000, 24000]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

        start = True

        for iter_num in range(1, args.max_iter + 1):
            model.train(True)
            classifier.train(True)
            optimizer.zero_grad()
            if iter_num % len_labeled == 0:
                iter_labeled = iter(dataset_loaders["train"])
            if iter_num % len_unlabeled == 0:
                iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

            data_labeled = iter_labeled.next()
            data_unlabeled = iter_unlabeled.next()

            img_labeled_q = data_labeled[0][0].to(device)
            img_labeled_k = data_labeled[0][1].to(device)
            label = data_labeled[1].to(device)

            img_unlabeled_q = data_unlabeled[0][0].to(device)
            img_unlabeled_k = data_unlabeled[0][1].to(device)
            label_in_unlabeldata = data_unlabeled[1].to(device)

            ## For Labeled Data
            PGC_logit_labeled, PGC_label_labeled, feat_labeled = model(img_labeled_q, img_labeled_k, label)
            out = classifier(feat_labeled)
            classifier_loss = criterions['CrossEntropy'](out, label)
            PGC_loss_labeled = criterions['KLDiv'](PGC_logit_labeled,
                                                   PGC_label_labeled)  # Contrastive loss for instances with the same labels

            ## For Unlabeled Data
            _, q_f_unlabeled = model.encoder_q(img_unlabeled_q)
            logit_unlabeled = classifier(q_f_unlabeled)

            prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
            confidence_unlabeled, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)
            PGC_logit_unlabeled, PGC_label_unlabeled, feat_unlabeled = model(img_unlabeled_q, img_unlabeled_k,
                                                                             predict_unlabeled)  # predict_unlabeled/pseudo_label
            PGC_loss_unlabeled = criterions['KLDiv'](PGC_logit_unlabeled, PGC_label_unlabeled)

            #compute Pseudo label acc
            if start:
                all_labels = label_in_unlabeldata.data.float()
                all_outputs = predict_unlabeled.data.float()
                start = False
            else:
                all_labels = torch.cat((all_labels, label_in_unlabeldata.data.float()),0)
                all_outputs = torch.cat((all_outputs, predict_unlabeled.data.float()),0)

            if iter_num % len_unlabeled == 0:
                pseudo_accuracy = torch.sum(all_outputs == all_labels).item() / float(all_labels.size()[0])
                print("iter_num:{}; Pseudo Label Acc{}".format(iter_num, pseudo_accuracy))
                start = True


            # prob_unlabeled_psuedo = torch.softmax(logit_unlabeled_psuedo.detach(), dim=-1)
            # confidence_unlabeled_psuedo, predict_unlabeled_psuedo = torch.max(prob_unlabeled_psuedo, dim=-1)

            total_loss = classifier_loss + PGC_loss_labeled + PGC_loss_unlabeled  # + classifier_unlabel_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            ## Calculate the training accuracy of current iteration
            if iter_num % 100 == 0:
                _, predict = torch.max(out, 1)
                hit_num = (predict == label).sum().item()
                sample_num = predict.size(0)
                print("iter_num: {0:2d}; current acc: {1:8.2f}".format(iter_num, hit_num / float(sample_num)))

            ## Show Loss in TensorBoard
            writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
            # writer.add_scalar('loss/classifier_loss', classifier_unlabel_loss, iter_num)
            # writer.add_scalar('loss/PGC_loss_labeled', PGC_loss_labeled, iter_num)
            # writer.add_scalar('loss/PGC_loss_unlabeled', PGC_loss_unlabeled, iter_num)
            writer.add_scalar('loss/total_loss', total_loss, iter_num)
            # print(iter_num % args.test_interval == 1 or iter_num == 500)
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

        # is_loop = is_pick_unlabeled_data(df_unlabeled_cluster)


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
    print(device)
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
    network, feature_dim = load_network('MOCOv2')
    model_moco = MOCOTuning(network=network, backbone='MOCOv2', queue_size=args.queue_size,
                            projector_dim=args.projector_dim, feature_dim=feature_dim,
                            class_num=args.class_num, momentum=args.momentum, pretrained=args.pretrained,
                            pretrained_path=args.pretrained_path).to(device)

    classifier_step2 = Classifier(feature_dim, args.class_num).to(device)  # classifier of step2

    # network for step 4
    net, feature_dim = load_network('resnet50')
    model_ce = net(projector_dim=1000, pretrained=True).to(device)
    classifier_ce = Classifier(feature_dim, args.class_num).to(device)  # classifier of step 4

    # Re-Initialize model, using pretrained resnet50 in step 4-6 for higher accuracy
    network, feature_dim = load_network(args.backbone)
    model = SelfTuning(network=network, backbone=args.backbone, queue_size=args.queue_size,
                       projector_dim=args.projector_dim, feature_dim=feature_dim,
                       class_num=args.class_num, momentum=args.momentum, pretrained=args.pretrained,
                       pretrained_path=args.pretrained_path).to(device)

    classifier = Classifier(feature_dim, args.class_num).to(device)  # classifier of SSL

    print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))

    ## Define Optimizer for optimize MOCO's  resnet50
    optimizer_moco = optim.SGD([
        {'params': model_moco.parameters()},
        {'params': classifier_step2.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000]
    scheduler_moco = torch.optim.lr_scheduler.MultiStepLR(optimizer_moco, milestones, gamma=0.1)

    ## Define Optimizer for optimize step4's resnet50
    optimizer_ce = optim.SGD([
        {'params': model_ce.parameters()},
        {'params': classifier_ce.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000]
    scheduler_ce = torch.optim.lr_scheduler.MultiStepLR(optimizer_ce, milestones, gamma=0.1)

    ## Define Optimizer for SSL
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # Train model
    train(args, model, model_ce, model_moco, classifier_step2, classifier, classifier_ce, dataset_loaders, optimizer_ce,
          optimizer_moco,
          scheduler_ce, scheduler_moco, device=device, writer=writer, model_path=model_path)


if __name__ == '__main__':
    main()
