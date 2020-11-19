import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import os
import time
import AverageMeter
import SERGCN
import resnet_roi
from sklearn.metrics import recall_score
from utils.eval import compute_map
from utils.logger import logger
from utils.epoch_utils import generate_dataloader
import logging

parser = argparse.ArgumentParser(description='PyTorch Social Relation')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--image-size', type=int, default=448, metavar='N',
                    help='the size of image (default: 448)')
parser.add_argument('--num-workers',  default=2, type=int,
                   help='number of load data workers (default: 2)')
parser.add_argument('--save-model', type=str, default='./Save_Model/',
                        help='where you save model')
parser.add_argument('--lr', type=float, default=0.00002,
                    help='fc layer learning rate (default: 0.00002)')
parser.add_argument('--max-epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--print-freq',  default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--manualSeed',  type=int, default=-1,
                     help='manual seed')
parser.add_argument('--time-steps', type=int, default=2, metavar='N',
                    help='the time steps (default: 2)')
parser.add_argument('--load-model', action='store_true', default=False,
                    help='whether load model or not')
parser.add_argument('--dataset', type=str, default='pipa_fine',
                    help='pipa_fine, pipa_coarse, pisc_fine, pisc_coarse')

parser.add_argument('--regenerate-roifeat', action='store_true', default=False)
parser.add_argument('--log-filename', type=str, default='log_info.txt')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
if args.manualSeed is None or args.manualSeed < 0:
    args.manualSeed = random.randint(1, 10000)
args.log_filename = args.log_filename.split('.')[0]+'_{}.txt'.format(args.manualSeed)

# FileHandler
file_handler = logging.FileHandler(args.log_filename)
file_handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s|%(filename)s[%(lineno)d]|%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if 'pipa' in args.dataset:
    args.images_root = '../../Dataset/PIPA_image'
    args.loss_weight = False
    args.max_person = 5
    if 'coarse' in args.dataset:
        args.train_file_pre = './relation_split/PIPA/PIPA_coarse_relation_train'
        args.valid_file_pre = './relation_split/PIPA/PIPA_coarse_relation_valid'
        args.test_file_pre = './relation_split/PIPA/PIPA_coarse_relation_test'
    else:
        args.train_file_pre = './relation_split/PIPA/PIPA_fine_relation_train'
        args.valid_file_pre = './relation_split/PIPA/PIPA_fine_relation_valid'
        args.test_file_pre = './relation_split/PIPA/PIPA_fine_relation_test'

elif 'pisc' in args.dataset:
    args.images_root = '../../Dataset/PISC_image'
    if 'coarse' in args.dataset:
        args.train_file_pre = './relation_split/PISC/PISC_coarse_relation_train'
        args.valid_file_pre = './relation_split/PISC/PISC_coarse_relation_valid'
        args.test_file_pre = './relation_split/PISC/PISC_coarse_relation_test'
        args.loss_weight = False
        args.max_person = 8
    else:
        args.train_file_pre = './relation_split/PISC/PISC_fine_relation_train'
        args.valid_file_pre = './relation_split/PISC/PISC_fine_relation_valid'
        args.test_file_pre = './relation_split/PISC/PISC_fine_relation_test'
        args.loss_weight = True
        args.max_person = 8
else:
    raise ValueError('Unknown dataset {}'.format(args.dataset))

if args.dataset == 'pipa_fine':
    args.num_classes = 16
elif args.dataset == 'pipa_coarse':
    args.num_classes = 5
elif args.dataset == 'pisc_fine':
    args.num_classes = 6
elif args.dataset == 'pisc_coarse':
    args.num_classes = 3
else:
    raise ValueError('Unknown dataset {}'.format(args.dataset))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(args.manualSeed)

logger.info(args)

class edge_loss(nn.Module):
    def __init__(self):
        super(edge_loss, self).__init__()
        if args.loss_weight:  # for PISC fine dataset
            self.criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weight).float())
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, scores, labels, masks):  # labels masks shape [batch_size, max_person, max_person]
        masks = masks.view(-1, 1).bool()
        masks = masks.detach()  # [batch_size*max_person*max_person, 1]
        labels = labels.view((-1, 1))[masks]  # [batch_size*max_person*max_person, 1]

        # scores shape [batch_size, max_person, max_person, num_classes]
        scores = scores.view(-1, args.num_classes)  # [batch_size*max_person*max_person, num_classes]
        scores = scores[masks.repeat(1, args.num_classes)].view(-1, args.num_classes)
        losses = self.criterion(scores, labels)

        return losses

def cal_acc(logits, labels, masks):
    labels_np = labels.data.cpu().long().numpy()
    masks_np = masks.data.cpu().long().numpy()
    count = np.sum(masks_np)
    acc_list = []
    all_logits_np = []

    logits_np = F.softmax(logits, dim=-1).data.cpu().numpy()  # [batch_size, max_person, max_person, num_classes]
    all_logits_np.append(logits_np)
    pred = np.argmax(logits_np, axis=3)  # [batch_size, max_person, max_person]
    res = (pred == labels_np)
    res = res * masks_np
    right_num = np.sum(res)
    acc_list.append(right_num * 1.0 / count)

    conf = logits_np[np.where(masks_np == 1)]
    pred_label = pred[np.where(masks_np == 1)]  # ndarray, size 1
    true_label = labels_np[np.where(masks_np == 1)]

    return acc_list, count, true_label, pred_label, conf

checkpoint_name = os.path.join(args.save_model, str(args.image_size) + args.dataset + '-checkpoint.txt')

def load_model(unload_model):
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
        logger.info(args.save_model,'is created!')
    if not os.path.exists(checkpoint_name):
        f = open(checkpoint_name, 'w')
        logger.info('checkpoint', 'is created!')

    start_index = 0
    with open(checkpoint_name, 'r') as fin:
        lines = fin.readlines()
        if len(lines) > 0:
            model_path, model_index = lines[0].split()
            logger.info('Resuming from {} with epoch {}'.format(model_path, model_index))
            if int(model_index) == 0:
                unload_model_dict = unload_model.state_dict()

                pretrained_dict = torch.load(os.path.join(args.save_model,model_path))

                pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in unload_model_dict and pretrained_dict[k].shape == unload_model_dict[k].shape )}           
                unload_model_dict.update(pretrained_dict)
                unload_model.load_state_dict(unload_model_dict)       
            else:
                unload_model.load_state_dict(torch.load(os.path.join(args.save_model, model_path)))
            
            start_index = int(model_index) + 1
    return start_index
 

def save_model(tosave_model, epoch):
    model_path = 'model-' + str(args.image_size) + '-' + args.dataset + '.pth'
    save_path = os.path.join(args.save_model, model_path)
    torch.save(tosave_model.module.state_dict(), save_path)
    with open(checkpoint_name, 'w') as fin:
        fin.write(model_path + ' ' + str(epoch) + '\n')

#dataset prepare
#---------------------------------
logger.info('Loading dataset...')
roi_net = resnet_roi.resnet101_roi()

if args.cuda:
    roi_net.cuda()
    roi_net = torch.nn.DataParallel(roi_net)

trainloader, class_weight, cls_num_list = generate_dataloader(roi_net, 'train', args)
validloader, _, _ = generate_dataloader(roi_net, 'valid', args)
testloader, _, _ = generate_dataloader(roi_net, 'test', args)

torch.cuda.empty_cache()

##Model prepare
logger.info("Loading model...")
SRModel = SERGCN.SERGCN(num_class=args.num_classes, hidden_dim=2048,
                  time_step=args.time_steps, node_num=args.max_person
                  )

total_param = 0
for param in SRModel.parameters():
    total_param += np.prod(list(param.data.size()))
logger.info("Model total parameters in SRModel is {}".format(total_param))


if args.load_model:
    start_epoch = load_model(SRModel)
else:
    start_epoch = 1

if args.cuda:
    SRModel.cuda()

SRModel = torch.nn.DataParallel(SRModel)

criterion = edge_loss()
ssl_loss_func = nn.BCELoss()

if args.cuda:
    criterion.cuda()

optimizer = optim.Adam(SRModel.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

def train_epoch(epoch):

    batch_time = AverageMeter.AverageMeter()
    data_time = AverageMeter.AverageMeter()
    losses = AverageMeter.AverageMeter()
    acces = AverageMeter.AverageMeter()

    SRModel.train()

    end_time = time.time()
    for batch_idx, (feat, union_feat, relation_half_mask, relation_id, full_mask) in enumerate(trainloader):
        data_time.update(time.time() - end_time)

        if args.cuda:
            feat, union_feat, relation_half_mask, relation_id, full_mask = \
                feat.cuda(), union_feat.cuda(), relation_half_mask.cuda(), relation_id.cuda(), full_mask.cuda()
        feat, union_feat, relation_half_mask, relation_id, full_mask = \
            Variable(feat), Variable(union_feat), Variable(relation_half_mask), Variable(relation_id), Variable(full_mask)

        optimizer.zero_grad()

        logits = SRModel(feat, union_feat, full_mask)

        loss = criterion(logits, relation_id, relation_half_mask)

        loss.backward()
        optimizer.step()
        losses.update(loss.cpu().data.numpy())

        #calcalate accuracy
        acc_list, count, _, _, _ = cal_acc(logits, relation_id, relation_half_mask)
        acces.update(acc_list[-1], count)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        acc_str = ''
        acc_str += ('%.3f ' % acces.avg)

        if batch_idx % args.print_freq == 0:
            logger.info('Epoch: [%d][%d/%d]  '
                 'Time %.3f (%.3f)\t'
                 'Data %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                 'pair %.1f\t'
                  % (epoch, batch_idx, len(trainloader),
                    batch_time.val,batch_time.avg, data_time.val,data_time.avg,
                    losses.val,losses.avg, count * 1.0) + acc_str)


def valid_test_epoch(epoch, loader, eval_type):

    batch_time = AverageMeter.AverageMeter()

    losses = AverageMeter.AverageMeter()
    acces = AverageMeter.AverageMeter()

    SRModel.eval()

    end_time = time.time()
    true_labels, pred_labels, confs = [], [], []
    for batch_idx, (feat, union_feat, relation_half_mask, relation_id, full_mask) in enumerate(loader):

        if args.cuda:
            feat, union_feat,  relation_half_mask, relation_id, full_mask = \
                feat.cuda(), union_feat.cuda(), relation_half_mask.cuda(), relation_id.cuda(), full_mask.cuda()
        feat, union_feat, relation_half_mask, relation_id, full_mask = \
            Variable(feat), Variable(union_feat), Variable(relation_half_mask), Variable(relation_id), Variable(full_mask)

        logits = SRModel(feat, union_feat, full_mask)

        loss = criterion(logits, relation_id, relation_half_mask)

        losses.update(loss.cpu().data.numpy())

        #calcalate accuracy
        acc_list, count, true_label, pred_label, conf = cal_acc(logits, relation_id, relation_half_mask)
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        confs.append(conf)
        acces.update(acc_list[-1], count)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        acc_str = ''
        acc_str += ('%.3f ' % acces.avg)

        if batch_idx % args.print_freq == 0:
            logger.info('Epoch: [%d][%d/%d]  '
                 'Time %.3f (%.3f)\t'
                 'Loss %.3f (%.3f)\t'
                  % (epoch, batch_idx, len(loader),
                    batch_time.val, batch_time.avg,
                    losses.val,losses.avg) + acc_str)

    acc_str = ''
    acc_str += ('%.3f ' % acces.avg)
    logger.info(eval_type + ": Acc " + acc_str + '\n')

    true_labels = np.concatenate(true_labels).reshape(-1)
    pred_labels = np.concatenate(pred_labels).reshape(-1)
    confs = np.concatenate(confs)

    recalls = recall_score(true_labels, pred_labels, average=None)
    logger.info(eval_type + ": Recall {} \n".format(recalls))

    mAP = compute_map(confs, true_labels)
    logger.info(eval_type + ": mAP {} \n".format(mAP))

    return acces, recalls, mAP, pred_labels, true_labels

logger.info('Start training...')
logger.info("Random Seed is {}".format(args.manualSeed))

best_test_result = 0
best_test_epoch = 0
best_test_recalls = []
best_test_other = 0

best_valid_result = 0
best_valid_epoch = 0
best_valid_recalls = []
best_valid_other = 0

if args.load_model:
    args.max_epochs = start_epoch + 1

for epoch in range(start_epoch, args.max_epochs):
    logger.info('Epoch: %d start!' % epoch)

    epoch_start = time.time()

    if args.load_model:
        valid_acces, _, valid_map, _, _ = valid_test_epoch(epoch, validloader, 'valid')
        test_acces, test_recalls, test_map, test_pred, test_true = valid_test_epoch(epoch, testloader, 'test')
    else:
        train_epoch(epoch)
        valid_acces, _, valid_map, _, _ = valid_test_epoch(epoch, validloader, 'valid')
        test_acces, test_recalls, test_map, test_pred, test_true = valid_test_epoch(epoch, testloader, 'test')

    if 'pipa' in args.dataset:
        test_result = test_acces.avg
        test_other = test_map
        valid_result = valid_acces.avg
        valid_other = valid_map
        scheduler_step = valid_acces.avg

    elif 'pisc' in args.dataset:
        test_result = test_map
        test_other = test_acces.avg
        valid_result = valid_map
        valid_other = valid_acces.avg
        scheduler_step = valid_map

    if test_result > best_test_result:
        best_test_result = test_result
        best_test_epoch = epoch
        best_test_recalls = test_recalls
        best_test_other = test_other
        save_model(SRModel, epoch)
    if valid_result > best_valid_result:
        best_valid_result = valid_result
        best_valid_epoch = epoch
        best_valid_recalls = test_recalls
        best_valid_other = test_other
        best_valid = test_result

    logger.info('Epoch {} time {}'.format(epoch, time.time()-epoch_start))
    scheduler.step(scheduler_step)

logger.info("Test set best-test result is {} best other is {} epoch {} best recalls is {}".format(
    best_test_result, best_test_other, best_test_epoch, best_test_recalls))
logger.info("Test set best-valid result is {} best other is {} epoch {} best recalls is {}".format(
    best_valid, best_valid_other, best_valid_epoch, best_valid_recalls))
