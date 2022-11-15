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
from torch.autograd import Variable

from models.classifier import Classifier
from models.method import SelfTuning
from tensorboardX import SummaryWriter
from src.utils import load_network, load_data
from src.utils import load_network, load_data

#from src.CompactBilinearPooling import CompactBilinearPooling
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

import cv2

from PIL import Image
from torchvision import models
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101
import torch.nn.functional as F


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
            feat,_ = model.inference(inputs)
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
    # for yangyang test
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
                feat,_ = model.inference(inputs[j]) 
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
	
    len_unlabeledori = len(dataset_loaders["unlabeled_train_ori"])##############
    iter_unlabeledori = iter(dataset_loaders["unlabeled_train_ori"])#############

    len_unlabeled = len(dataset_loaders["unlabeled_train"])##############
    iter_unlabeled = iter(dataset_loaders["unlabeled_train"])#############

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    best_acc = 0.0
    best_model = None

    for iter_num in range(1, args.max_iter + 1):
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
        if iter_num % len_unlabeledori == 0:##########
            iter_unlabeledori = iter(dataset_loaders["unlabeled_train_ori"])#########
        if iter_num % len_unlabeled == 0:##########
            iter_unlabeled = iter(dataset_loaders["unlabeled_train"])#########
        

        data_labeled = iter_labeled.next()
        data_unlabeledori = iter_unlabeledori.next()##########
        data_unlabeled = iter_unlabeled.next()##########

        img_labeled_q = data_labeled[0][0].to(device)
        img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)

        img_unlabeled_qori = data_unlabeledori[0].to(device)######
        img_unlabeled_kori = data_unlabeledori[1].to(device)#########
		
        img_unlabeled_q = data_unlabeled[0][0].to(device)######
        img_unlabeled_k = data_unlabeled[0][1].to(device)#########

    
        ## For Labeled Data
        PGC_logit_labeled, PGC_label_labeled, feat_labeled, featmap_q, featmap_k, featcov16, bp_out_feat = model(img_labeled_q, img_labeled_k, label)   
		
		
       
        
        out = classifier(bp_out_feat.cuda())#bp_out_feat.cuda()
        
        classifier_loss = criterions['CrossEntropy'](out, label)
        PGC_loss_labeled = criterions['KLDiv'](PGC_logit_labeled, PGC_label_labeled)
		
        '''
        #使用CBP包
        cbp_dim = 2000
        mcb = CompactBilinearPooling(512, 16, cbp_dim).cuda()
        N,C1,H,W = featcov16.shape
        _,C2,_,_ = featcov512.shape
        feat_part = featcov16.clone().permute(0,2,3,1).contiguous().view(-1,C1)
        feat_whole = featcov512.clone().permute(0,2,3,1).contiguous().view(-1,C2)
        
        bp_out_feat = mcb(feat_whole.cuda(), feat_part.cuda())
        bp_out_feat = bp_out_feat.view(N,H,W,cbp_dim).permute(0,3,1,2).contiguous()
        bp_out_feat = bp_out_feat.view(N,cbp_dim,-1).sum(-1)
       
        cbp_norm = nn.BatchNorm1d(cbp_dim).to(device)
        cbp_norm.bias.requires_grad_(False)
        nn.init.constant_(cbp_norm.weight, 1)
        nn.init.constant_(cbp_norm.bias, 0)
        bp_out_feat = cbp_norm(bp_out_feat)
        '''
        #自己写BP
        '''
        feat_matrix = torch.zeros(featcov16.size(0),16,512)
        avgpool = nn.AvgPool2d(7, stride=1)
        bp_dim = feat_matrix.size(1)*feat_matrix.size(2)
        for i in range(16):
            matrix = featcov16[:,i,:,:]
            matrix = matrix[:,None,:,:]
            matrix = matrix.repeat(1,512,1,1)
            PFM = featcov512*matrix#Batch*512*7*7
            aa =avgpool(PFM)#Batch*512*1*1
            feat_matrix[:,i,:] = aa.view(aa.size(0), -1)
        
           
        #feat_matrix:batch*16*512
        bp_out_feat= feat_matrix.view(feat_matrix.size(0), -1)
        #bp_norm = nn.BatchNorm1d(bp_dim).to(device)
        #bp_norm.bias.requires_grad_(False)
        #nn.init.constant_(bp_norm.weight, 1)
        #nn.init.constant_(bp_norm.bias, 0)
        #bp_out_feat = bp_norm(bp_out_feat.cuda())
        '''
     
		
	
        
        
        
        '''
        ## For Unlabeled Data
        q_c_unlabeledori, q_f_unlabeledori, featmap_unlabelori = model.encoder_q(img_unlabeled_qori)
        q_c_unlabeled, q_f_unlabeled, _ = model.encoder_q(img_unlabeled_q)#featmap_unlabel
        ##featmap_unlabelori = torch.mean(featmap_unlabelori, dim=1)
        ##featmap_unlabel = torch.mean(featmap_unlabel, dim=1)
        
        #if featmap_unlabelori.size(0) == featmap_unlabel.size(0):
            #loss_u = F.kl_div(featmap_unlabel.softmax(dim=-1).log(), featmap_unlabelori.softmax(dim=-1), reduction='sum')
        #lossmse = torch.nn.MSELoss()
        #param1 = torch.autograd.Variable(featmap_unlabel.softmax(dim=-1))
        #param2 = torch.autograd.Variable(featmap_unlabel.softmax(dim=-1))
        #if featmap_unlabelori.size(0) == featmap_unlabel.size(0):
            #loss_u = lossmse(param1, param2)
	    
		
        #q_c_unlabeled, q_f_unlabeled = model.encoder_q(img_unlabeled_q)
        logit_unlabeled = classifier(q_f_unlabeled)
        prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
        confidence_unlabeled, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)
        PGC_logit_unlabeled, PGC_label_unlabeled, feat_unlabeled, featmap_unlabel_q, featmap_unlabel_k = model(img_unlabeled_q, img_unlabeled_k, predict_unlabeled)
        PGC_loss_unlabeled = criterions['KLDiv'](PGC_logit_unlabeled, PGC_label_unlabeled)
        '''
		
        
        methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
        network, feature_dim = load_network(args.backbone)
        modelcam = network()
        ####model = models.resnet50(pretrained=True)
        target_layers = [modelcam.layer4[-1]]
        input_tensor_q = img_unlabeled_q  ###img_labeled_q
        input_tensor_k = img_unlabeled_k
        input_tensor_labeled_q = img_labeled_q  ###img_labeled_q
        input_tensor_labeled_k = img_labeled_k
        target_category = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method] 
           
        #labeled
        with cam_algorithm(model=modelcam,
                           target_layers=target_layers,
                           use_cuda=args.use_cuda) as cam:
        
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 24
        
            grayscale_cam, weights = cam(input_tensor=input_tensor_labeled_q,
                                target_category=target_category,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)#grayscale_cam:batch*224*224
            
            
            
            '''
            #求出权重w
            
            weights = torch.from_numpy(weights)
            weights = weights.to(device)
            weights = weights[:,:,None, None]
            weights = weights.repeat(1,1,7,7)#batch*2048*7*7
            #得到热力图的卷积特征作为target
            #yy, _ = torch.max(featmap_q, dim=1)
            #print(yy)
            featmapcam = featmap_q*weights #featmap用于label，而featmap_unlabel用于unlabel.  featmap_q: batchsize*2048*7*7
            featmapcam = torch.sum(featmapcam, dim=1)
            relu = nn.ReLU(inplace=True)
            featmapcam = relu(featmapcam)#batch*7*7
            #print(featmapcam)
			
            #预测分布
            predictcam, _ = torch.max(featcov16, dim=1)#batch*7*7
            predictcam = predictcam.to(device)
            '''
            img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
            img,_ = torch.max(img, axis=3)#batch*H*W
            img = img.cpu().detach().numpy()
            
            #img = np.max(img, axis=3)#batch*H*W
	
	
            for i in range(len(img)):
                img[i,:,:] = img[i,:,:] - np.min(img[i,:,:])
                img[i,:,:] = img[i,:,:] / (1e-7 + np.max(img[i,:,:]))
           
            img_torch = torch.from_numpy(img) 
            predict_cam = np.ones((img_torch.size(0),224,224))
            for i in range(img_torch.size(0)):
                one_img = img_torch[i,:,:]#H*W
                one_img = one_img.cpu().detach().numpy()
                predict_cam[i,:,:] = cv2.resize(one_img, (224,224))#H*W
            predict_cam = torch.from_numpy(predict_cam)
            grayscale_cam = torch.from_numpy(grayscale_cam)
            #print(grayscale_cam.size())
            #print(predict_cam.size())

            #使用KL
            #softmax的temperature parameter
            #t = 4
            #grayscale_cam = (grayscale_cam/t).float()
            #predict_cam = (predict_cam/t).float()
            #loss_cam_labeled_q = F.kl_div(grayscale_cam.softmax(dim=-1).log(), predict_cam.softmax(dim=-1), reduction='sum')#第一个是预测分布featmapcam，第二个是真实分布predictcam
            #使用MSE
            loss_fn1 = nn.MSELoss()#reduction='none'
            loss_cam_labeled_q = loss_fn1(grayscale_cam.float(), predict_cam.float())
          
       
        #print(loss_cam_labeled_q)
        total_loss = classifier_loss + 0.001 * loss_cam_labeled_q#classifier_loss + 0.001 * loss_kl_labeled_q#+ 0.01 * bp_loss#classifier_loss + 0.001 * loss_kl_labeled_q #classifier_loss + 0.001*loss_kl_labeled_q #0.001*bp_loss+loss_kl_labeled_k # loss_kl_q + loss_kl_k + PGC_loss_labeled + PGC_loss_unlabeled
        total_loss.backward(retain_graph=True)#retain_graph=True
        optimizer.step()
        scheduler.step()


        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)#####################################################out!!
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))

        ## Show Loss in TensorBoard
        writer.add_scalar('loss/cam_loss', loss_cam_labeled_q, iter_num)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
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
    '''
    for iter_num in range(1, args.max_iter + 1):
        model.train(True)
        classifier.train(True)
        
        #bottom1 = Variable(featcov512).cuda()#featcov512,
        #bottom2 = Variable(featcov16).cuda()
        feat_labeled1 = feat_labeled[:,:,None, None]
        feat_labeled1 = feat_labeled1.repeat(1,1,7,7)
        bottom1 = Variable(feat_labeled1).cuda()#featcov512,
        bottom2 = Variable(featcov16).cuda()
        layer = CompactBilinearPooling(2048, 16, 8000)#512
        layer.cuda()
        layer.train()
        bp_out_feat = layer(bottom1, bottom2)
        classifier_bp = Classifier(8000, args.class_num).to(device)
        bp_out = classifier_bp(bp_out_feat)
        bp_loss = criterions['CrossEntropy'](bp_out, label)
        
		

		
        total_loss = classifier_loss#classifier_loss + 0.001*loss_kl_labeled_q #0.001*bp_loss+loss_kl_labeled_k # loss_kl_q + loss_kl_k + PGC_loss_labeled + PGC_loss_unlabeled
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
        #writer.add_scalar('loss/cam_loss', loss_kl, iter_num)
        writer.add_scalar('loss/bp_loss', bp_loss, iter_num)
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
    '''
    print("best acc: %.4f" % (best_acc))
    torch.save(best_model, model_path)
    print("The best model has been saved in ", model_path)

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--label_ratio', type=int, default=15)
    parser.add_argument('--logdir', type=str, default='../vis/')
    parser.add_argument('--lr', type=float, default='0.001')#0.001
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
    parser.add_argument('--max_iter', type=float, default=27005)#27005
    parser.add_argument('--test_interval', type=float, default=3000)
    parser.add_argument("--pretrained", action="store_true", help="use the pre-trained model")
    parser.add_argument("--pretrained_path", type=str, default='~/.torch/models/moco_v2_800ep_pretrain.pth.tar')
    ##cam
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/test.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    
    ## Only for Cifar100
    parser.add_argument("--expand_label", action="store_true", help="expand label to fit eval steps")
    parser.add_argument('--num_labeled', type=int, default=0, help='number of labeled data')
    configs = parser.parse_args()
    configs.use_cuda = configs.use_cuda and torch.cuda.is_available()
    if configs.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
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
    #Haiming provide----
    logdir = os.path.join(args.logdir, model_name)
    method_name = 'self_tuning_logConfid'
    method_name += '_qdim' + str(args.projector_dim)
    #if args.output_index:
        #method_name += '_trackPLStats'
    #if args.use_identity_projector:
        #method_name += '_noPrjctr'
    #if args.weak_strong_aug_l
        #method_name += '_augWSL'
    #if args.weak_strong_aug:
        #method_name += '_augWSL'
    #if not args.pretrained:
        #method_name += '_notPretrained'
    method_name += '_seed' + str(args.seed)
    #----
    logdir = os.path.join(args.logdir, method_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    model_path = os.path.join(logdir, "%s_best.pkl" % (model_name))

    # Initialize model
    network, feature_dim = load_network(args.backbone)
    model = SelfTuning(network=network, backbone=args.backbone, queue_size=args.queue_size, projector_dim=args.projector_dim, feature_dim=feature_dim,
                       class_num=args.class_num, momentum=args.momentum, pretrained=args.pretrained, pretrained_path=args.pretrained_path).to(device)
    classifier = Classifier(8192, args.class_num).to(device)#######################feature_dim

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
    
    # cam
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
		 
    model1 = network()
    target_category = None
    ####model = models.resnet50(pretrained=True)
    target_layers = [model1.layer4[-1]]
    #rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]#读取图片，1表示彩色，：：-1表示倒序
	
    rgb_img = cv2.imread('./examples/test.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img0 = rgb_img[63:287,63:287,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    #rgb_img = Image.fromarray(rgb_img)
    input_tensor0 = preprocess_image(rgb_img0,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
 
    _, featcov16 = model.inference(input_tensor0.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16

    img = img.cpu().detach().numpy()
    
    img = np.max(img, axis=2)
	
	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
	
    img0 = cv2.resize(img, (224,224))#H*W
	
    #------------------1
    rgb_img = cv2.imread('./examples/test1.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img1 = rgb_img[76:300,138:362,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor1 = preprocess_image(rgb_img1,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor1.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img1 = cv2.resize(img, (224,224))#H*W
    #------------------2
    rgb_img = cv2.imread('./examples/test2.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img2 = rgb_img[0:224,146:370,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor2 = preprocess_image(rgb_img2,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor2.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img2 = cv2.resize(img, (224,224))#H*W
    #------------------3
    rgb_img = cv2.imread('./examples/test3.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img3 = rgb_img[70:294,13:237,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor3 = preprocess_image(rgb_img3,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor3.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img3 = cv2.resize(img, (224,224))#H*W
    #------------------4
    rgb_img = cv2.imread('./examples/test4.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img4 = rgb_img[110:334,55:279,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor4 = preprocess_image(rgb_img4,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor4.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img4 = cv2.resize(img, (224,224))#H*W
    #------------------5
    rgb_img = cv2.imread('./examples/test5.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img5 = rgb_img[0:224,125:349,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor5 = preprocess_image(rgb_img5,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor5.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img5 = cv2.resize(img, (224,224))#H*W
    #------------------6
    rgb_img = cv2.imread('./examples/test6.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img6 = rgb_img[41:265,26:250,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor6 = preprocess_image(rgb_img6,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor6.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img6 = cv2.resize(img, (224,224))#H*W
    #------------------7
    rgb_img = cv2.imread('./examples/test7.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img7 = rgb_img[88:312,138:362,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor7 = preprocess_image(rgb_img7,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor7.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img7 = cv2.resize(img, (224,224))#H*W
    #------------------8
    rgb_img = cv2.imread('./examples/test8.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img8 = rgb_img[10:234,20:244,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor8 = preprocess_image(rgb_img8,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor8.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img8 = cv2.resize(img, (224,224))#H*W
    #------------------9
    rgb_img = cv2.imread('./examples/test9.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img9 = rgb_img[109:333,138:362,0:3]#test9[109:333,138:362,0:3]test8[10:234,20:244,0:3]test7[88:312,138:362,0:3]test6[41:265,26:250,0:3]test5[0:224,125:349,0:3]test4[110:334,55:279,0:3]test3:[70:294,13:237,0:3]test2[0:224,146:370,0:3]test[63:287,63:287,0:3]test1[76:300,138:362,0:3]
    input_tensor9 = preprocess_image(rgb_img9,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])#1*3*224*224
    _, featcov16 = model.inference(input_tensor9.to(device)) #Batch*16*H*W
    img = featcov16.clone().permute(0,2,3,1).contiguous()#batch*H*W*C
    img = img[0,:,:,:]#H*W*16
    img = img.cpu().detach().numpy()  
    img = np.max(img, axis=2)	
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    img9 = cv2.resize(img, (224,224))#H*W

	 
    
    
    
    
    
    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]

    with cam_algorithm(model=model1,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:
        
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        
        grayscale_cam0, weights = cam(input_tensor=input_tensor0,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224
        
        
        # Here grayscale_cam has only one image in the batch
        grayscale_cam0 = grayscale_cam0[0, :]#224*224
        
        cam_image0g = show_cam_on_image(rgb_img0, grayscale_cam0, use_rgb=True)
        cam_image0p = show_cam_on_image(rgb_img0, img0, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image0g = cv2.cvtColor(cam_image0g, cv2.COLOR_RGB2BGR)
        cam_image0p = cv2.cvtColor(cam_image0p, cv2.COLOR_RGB2BGR)
        #--------------------1	
        grayscale_cam1, weights = cam(input_tensor=input_tensor1,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam1 = grayscale_cam1[0, :]#224*224
        cam_image1g = show_cam_on_image(rgb_img1, grayscale_cam1, use_rgb=True)
        cam_image1p = show_cam_on_image(rgb_img1, img1, use_rgb=True)
        cam_image1g = cv2.cvtColor(cam_image1g, cv2.COLOR_RGB2BGR)
        cam_image1p = cv2.cvtColor(cam_image1p, cv2.COLOR_RGB2BGR)
        #--------------------2
        grayscale_cam2, weights = cam(input_tensor=input_tensor2,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam2 = grayscale_cam2[0, :]#224*224
        cam_image2g = show_cam_on_image(rgb_img2, grayscale_cam2, use_rgb=True)
        cam_image2p = show_cam_on_image(rgb_img2, img2, use_rgb=True)
        cam_image2g = cv2.cvtColor(cam_image2g, cv2.COLOR_RGB2BGR)
        cam_image2p = cv2.cvtColor(cam_image2p, cv2.COLOR_RGB2BGR)
        #--------------------3	
        grayscale_cam3, weights = cam(input_tensor=input_tensor3,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam3 = grayscale_cam3[0, :]#224*224
        cam_image3g = show_cam_on_image(rgb_img3, grayscale_cam3, use_rgb=True)
        cam_image3p = show_cam_on_image(rgb_img3, img3, use_rgb=True)
        cam_image3g = cv2.cvtColor(cam_image3g, cv2.COLOR_RGB2BGR)
        cam_image3p = cv2.cvtColor(cam_image3p, cv2.COLOR_RGB2BGR)
        #--------------------4
        grayscale_cam4, weights = cam(input_tensor=input_tensor4,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam4 = grayscale_cam4[0, :]#224*224
        cam_image4g = show_cam_on_image(rgb_img4, grayscale_cam4, use_rgb=True)
        cam_image4p = show_cam_on_image(rgb_img4, img4, use_rgb=True)
        cam_image4g = cv2.cvtColor(cam_image4g, cv2.COLOR_RGB2BGR)
        cam_image4p = cv2.cvtColor(cam_image4p, cv2.COLOR_RGB2BGR)
        #--------------------5	
        grayscale_cam5, weights = cam(input_tensor=input_tensor5,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam5 = grayscale_cam5[0, :]#224*224
        cam_image5g = show_cam_on_image(rgb_img5, grayscale_cam5, use_rgb=True)
        cam_image5p = show_cam_on_image(rgb_img5, img5, use_rgb=True)
        cam_image5g = cv2.cvtColor(cam_image5g, cv2.COLOR_RGB2BGR)
        cam_image5p = cv2.cvtColor(cam_image5p, cv2.COLOR_RGB2BGR)
        #--------------------6
        grayscale_cam6, weights = cam(input_tensor=input_tensor6,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam6 = grayscale_cam6[0, :]#224*224
        cam_image6g = show_cam_on_image(rgb_img6, grayscale_cam6, use_rgb=True)
        cam_image6p = show_cam_on_image(rgb_img6, img6, use_rgb=True)
        cam_image6g = cv2.cvtColor(cam_image6g, cv2.COLOR_RGB2BGR)
        cam_image6p = cv2.cvtColor(cam_image6p, cv2.COLOR_RGB2BGR)
        #--------------------7	
        grayscale_cam7, weights = cam(input_tensor=input_tensor7,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam7 = grayscale_cam7[0, :]#224*224
        cam_image7g = show_cam_on_image(rgb_img7, grayscale_cam7, use_rgb=True)
        cam_image7p = show_cam_on_image(rgb_img7, img7, use_rgb=True)
        cam_image7g = cv2.cvtColor(cam_image7g, cv2.COLOR_RGB2BGR)
        cam_image7p = cv2.cvtColor(cam_image7p, cv2.COLOR_RGB2BGR)
        #--------------------8
        grayscale_cam8, weights = cam(input_tensor=input_tensor8,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam8 = grayscale_cam8[0, :]#224*224
        cam_image8g = show_cam_on_image(rgb_img8, grayscale_cam8, use_rgb=True)
        cam_image8p = show_cam_on_image(rgb_img8, img8, use_rgb=True)
        cam_image8g = cv2.cvtColor(cam_image8g, cv2.COLOR_RGB2BGR)
        cam_image8p = cv2.cvtColor(cam_image8p, cv2.COLOR_RGB2BGR)
        #--------------------9	
        grayscale_cam9, weights = cam(input_tensor=input_tensor9,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)#grayscale_cam: batch*224*224

        grayscale_cam9 = grayscale_cam9[0, :]#224*224
        cam_image9g = show_cam_on_image(rgb_img9, grayscale_cam9, use_rgb=True)
        cam_image9p = show_cam_on_image(rgb_img9, img9, use_rgb=True)
        cam_image9g = cv2.cvtColor(cam_image9g, cv2.COLOR_RGB2BGR)
        cam_image9p = cv2.cvtColor(cam_image9p, cv2.COLOR_RGB2BGR)
		
		
    #gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    #gb = gb_model(input_tensor, target_category=target_category)

    #cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    #cam_gb = deprocess_image(cam_mask * gb)
    #gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam0g.jpg', cam_image0g)
    cv2.imwrite(f'{args.method}_cam0p.jpg', cam_image0p)
    #----------------1
    cv2.imwrite(f'{args.method}_cam1g.jpg', cam_image1g)
    cv2.imwrite(f'{args.method}_cam1p.jpg', cam_image1p)
    #----------------2
    cv2.imwrite(f'{args.method}_cam2g.jpg', cam_image2g)
    cv2.imwrite(f'{args.method}_cam2p.jpg', cam_image2p)
    #----------------3
    cv2.imwrite(f'{args.method}_cam3g.jpg', cam_image3g)
    cv2.imwrite(f'{args.method}_cam3p.jpg', cam_image3p)
    #----------------4
    cv2.imwrite(f'{args.method}_cam4g.jpg', cam_image4g)
    cv2.imwrite(f'{args.method}_cam4p.jpg', cam_image4p)
    #----------------5
    cv2.imwrite(f'{args.method}_cam5g.jpg', cam_image5g)
    cv2.imwrite(f'{args.method}_cam5p.jpg', cam_image5p)
    #----------------6
    cv2.imwrite(f'{args.method}_cam6g.jpg', cam_image6g)
    cv2.imwrite(f'{args.method}_cam6p.jpg', cam_image6p)
    #----------------7
    cv2.imwrite(f'{args.method}_cam7g.jpg', cam_image7g)
    cv2.imwrite(f'{args.method}_cam7p.jpg', cam_image7p)
    #----------------8
    cv2.imwrite(f'{args.method}_cam8g.jpg', cam_image8g)
    cv2.imwrite(f'{args.method}_cam8p.jpg', cam_image8p)
    #----------------9
    cv2.imwrite(f'{args.method}_cam9g.jpg', cam_image9g)
    cv2.imwrite(f'{args.method}_cam9p.jpg', cam_image9p)
    
    #cv2.imwrite(f'{args.method}_gb.jpg', gb)
    #cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
   
if __name__ == '__main__':
    main()
