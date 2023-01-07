from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import optim, nn

from src.main import accuracy_top1
from src.utils import load_network
from models.classifier import Classifier

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
    dataframe.to_csv('unlabeled_cluster' + str(args.class_num) + str(args.label_ratio) + str(args.confidence) + '.csv', index=False)
    return dataframe


def select_unlabel_data(args):
    unlabel_cluster = pd.read_csv('unlabeled_cluster' + str(args.class_num) + str(args.label_ratio) + str(args.confidence) + '.csv')
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
    dataframe.to_csv('select_suitable_unlabeled_data' + str(args.class_num) + str(args.label_ratio) + str(args.confidence) + '.csv',
                     index=False)
    return dataframe


def process_unlabel_data_step2(args,device,labeled_data,unlabeled_data,model_moco):
    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}
    net, feature_dim = load_network('resnet50')
    model_ce = net(projector_dim=1000, pretrained=True).to(device)
    classifier_ce = Classifier(feature_dim, args.class_num).to(device)

    ## Define Optimizer for optimize step4's resnet50
    optimizer_ce = optim.SGD([
        {'params': model_ce.parameters()},
        {'params': classifier_ce.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000]
    scheduler_ce = torch.optim.lr_scheduler.MultiStepLR(optimizer_ce, milestones, gamma=0.1)
    train_model_CE(classifier_ce, criterions, labeled_data, device, model_ce, optimizer_ce, scheduler_ce)
    process_unlabel_data_step3(args, classifier_ce, unlabeled_data, device, model_moco, model_ce)

def process_unlabel_data_step3(args, classifier_ce, dataset_loader, device, model_moco, model_ce):  # first mode for cycle, second for psuedo
    print('step3 starts')
    data = np.zeros((1, 2048))
    label = np.zeros(1)
    pseudo_label = np.zeros(1)
    confidence = np.zeros(1)
    arr_path = ['first']
    model_moco.encoder_q.eval()
    model_ce.eval()
    classifier_ce.eval()

    with torch.no_grad():
        for i, (images, target, path) in enumerate(dataset_loader):
            images = images[0].to(device)
            # img_unlabeled_k = data_unlabeled[0][1].to(device)

            # arrange pseudo label
            _, q_f_unlabeled = model_moco.encoder_q(images)  # feat for retrieval using MOCOv2
            _, unlabeled_feat = model_ce(images)  # feat for arrange psuedo label using Resnet50(pretrained=True)
            logit_unlabeled = classifier_ce(unlabeled_feat)
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
            'select_suitable_unlabeled_data' + str(args.class_num) + str(args.label_ratio) + str(args.confidence) +'.csv')
        # with open('./StanfordCars/image_list/right_psuedo_train.txt', 'w') as f:
        # for line in select_suitable_unlabeled_data.values:
        # f.write((str(line[0][9:])) + '\t' + str(int(line[2])) + '\n')
    return df_unlabeled_cluster, df_select_unlabel_data

def train_model_CE(classifier_ce, criterions, dataset_loader, device, model_ce, optimizer_ce, scheduler_ce):
    # step4: Using labeled data (CE loss) to fine-tuning MOCOv2
    print('step4 starts')
    len_labeled = len(dataset_loader)
    iter_labeled = iter(dataset_loader)
    for iter_num in range(1, 12000 + 1):  # args.max_iter + 1
        model_ce.train(True)
        classifier_ce.train(True)
        optimizer_ce.zero_grad()

        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loader)

        data_labeled = iter_labeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        # img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)

        ## For Labeled Data
        _, feat_labeled = model_ce(img_labeled_q)
        out = classifier_ce(feat_labeled)
        classifier_loss = criterions['CrossEntropy'](out, label)

        total_loss = classifier_loss
        total_loss.backward()
        optimizer_ce.step()
        scheduler_ce.step()

        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {0:2d}; current acc: {1:8.2f}".format(iter_num, hit_num / float(sample_num)))