from utils.logger import logger
import SRDataset
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import random
import logging
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torchvision.transforms as transforms
from RIG import SSLModel as model
import AverageMeter
import time
import os
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='PyTorch Social Relation')
parser.add_argument('--log-filename', type=str, default='log_info.txt')
parser.add_argument('--manualSeed',  type=int, default=1, help='manual seed')
parser.add_argument('--batch-size',  type=int, default=48, help='manual seed')
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--num-workers',  default=16, type=int)
parser.add_argument('--lr', type=float, default=0.00001)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if args.manualSeed is None or args.manualSeed < 0:
    args.manualSeed = random.randint(1, 10000)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(args.manualSeed)

# FileHandler
file_handler = logging.FileHandler(args.log_filename)
file_handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s|%(filename)s[%(lineno)d]|%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(args)

images_root = '../../Dataset'
images_list_path = '../20W_img_list.txt'
img_pos_dict_path = '../20W_scene_pos_ssl.json'
img_neg_dict_path = '../20W_scene_neg_ssl.json'


img_count = 0
with open(images_list_path, 'r') as fin:
    for line in fin:
        img_count += 1
train_idx, test_idx = train_test_split(np.arange(img_count), test_size=0.2, random_state=args.manualSeed)
cache_size = 256 * 2
args.image_size = 448
transform_train = transforms.Compose([
        transforms.Resize((cache_size, cache_size)),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

train_data = SRDataset.SceneDataset(image_dir=images_root, images_list_path=images_list_path,
                                  img_pos_dict_path=img_pos_dict_path,img_neg_dict_path = img_neg_dict_path,
                                  idxs=train_idx, input_transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers,
                                             worker_init_fn=np.random.seed(args.manualSeed))
test_data = SRDataset.SceneDataset(image_dir=images_root, images_list_path=images_list_path,
                                  img_pos_dict_path=img_pos_dict_path,img_neg_dict_path = img_neg_dict_path,
                                  idxs=test_idx, input_transform=transform_test)
testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers,
                                             worker_init_fn=np.random.seed(args.manualSeed))

torch.cuda.empty_cache()

model_file = '../resnet50_places365.pth.tar'  # open sourced pre-trained model
resnet_model = models.__dict__['resnet50'](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
resnet_model.load_state_dict(state_dict)

SSL_model = model(resnet_model)

if args.cuda:
    SSL_model.cuda()
SSL_model = torch.nn.DataParallel(SSL_model)

total_param = 0
for param in SSL_model.parameters():
    total_param += np.prod(list(param.data.size()))
logger.info("Model total parameters in SRModel is {}".format(total_param))

optimizer = torch.optim.Adam(SSL_model.parameters(), lr=args.lr)
criterion = nn.BCELoss(reduction='mean')

# training
def train_epoch(epoch):
    losses = AverageMeter.AverageMeter()
    acces = AverageMeter.AverageMeter()
    auces = AverageMeter.AverageMeter()

    SSL_model.train()
    scores_list, labels_list = [], []
    for batch_idx, (img, img_pos, img_neg) in enumerate(trainloader):

        if args.cuda:
            img, img_pos, img_neg = img.cuda(), img_pos.cuda(), img_neg.cuda()
        img, img_pos, img_neg = Variable(img), Variable(img_pos), Variable(img_neg)

        optimizer.zero_grad()

        score_pos, score_neg = SSL_model(img, img_pos, img_neg)
        label_pos = torch.ones_like(score_pos)
        label_neg = torch.zeros_like(score_neg)

        scores = torch.cat([score_pos, score_neg])
        labels = torch.cat([label_pos, label_neg])

        loss = criterion(scores, labels)

        loss.backward()
        optimizer.step()
        losses.update(loss.cpu().data.numpy())

        scores_np = scores.data.cpu().numpy()
        labels_np = labels.data.cpu().long().numpy()
        scores_list.append(scores_np)
        labels_list.append(labels_np)

        pred_np = np.zeros_like(labels_np)
        pred_np[scores_np >= np.median(scores_np)] = 1
        acces.update(np.sum(pred_np == labels_np)/len(labels_np), len(labels_np))

        precision, recall, thresholds = metrics.precision_recall_curve(labels_np, scores_np)
        auc = metrics.auc(recall, precision)
        auces.update(auc)

        if batch_idx % 50 == 0:
            logger.info('Epoch: [%d][%d/%d]  ' 
                 'Loss %.3f (%.3f)\t '
                 'acc %.3f (%.3f)\t'
                 'auc %.3f (%.3f)\t '
                 'score max %.3f min %.3f median %.3f\t'
                  % (epoch, batch_idx, len(trainloader),
                    losses.val, losses.avg,
                     acces.val, acces.avg,
                     auces.val, auces.avg,
                     np.max(scores_np), np.min(scores_np), np.median(scores_np)))

    scores = np.concatenate(scores_list)
    labels = np.concatenate(labels_list)
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    auc = metrics.auc(recall, precision)

    pred_np = np.zeros_like(labels)
    pred_np[scores >= np.median(scores)] = 1
    acc = np.sum(pred_np == labels) / len(labels)
    logger.info('Epoch {} acc {:.4f} auc {:.4f}'.format(epoch, acc, auc))


def test_epoch(epoch):
    with torch.no_grad():
        losses = AverageMeter.AverageMeter()
        acces = AverageMeter.AverageMeter()
        auces = AverageMeter.AverageMeter()

        SSL_model.eval()
        scores_list, labels_list = [], []
        for batch_idx, (img, img_pos, img_neg) in enumerate(testloader):
            if args.cuda:
                img, img_pos, img_neg = img.cuda(), img_pos.cuda(), img_neg.cuda()
            img, img_pos, img_neg = Variable(img), Variable(img_pos), Variable(img_neg)

            score_pos, score_neg = SSL_model(img, img_pos, img_neg)
            label_pos = torch.ones_like(score_pos)
            label_neg = torch.zeros_like(score_neg)

            scores = torch.cat([score_pos, score_neg])
            labels = torch.cat([label_pos, label_neg])

            loss = criterion(scores, labels)

            losses.update(loss.cpu().data.numpy())

            scores_np = scores.data.cpu().numpy()
            labels_np = labels.data.cpu().long().numpy()
            scores_list.append(scores_np)
            labels_list.append(labels_np)

            pred_np = np.zeros_like(labels_np)
            pred_np[scores_np >= np.median(scores_np)] = 1
            acces.update(np.sum(pred_np == labels_np)/len(labels_np), len(labels_np))

            precision, recall, thresholds = metrics.precision_recall_curve(labels_np, scores_np)
            auc = metrics.auc(recall, precision)
            auces.update(auc)

            if batch_idx % 50 == 0:
                logger.info('Test Epoch: [%d][%d/%d]  ' 
                     'Loss %.3f (%.3f)\t'
                     'acc %.3f (%.3f)\t'
                     'auc %.3f (%.3f)\t'
                     'score max %.3f min %.3f median %.3f\t'
                      % (epoch, batch_idx, len(testloader),
                        losses.val, losses.avg,
                         acces.val, acces.avg,
                         auces.val, auces.avg,
                         np.max(scores_np), np.min(scores_np), np.median(scores_np)))

        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
        auc = metrics.auc(recall, precision)

        pred_np = np.zeros_like(labels)
        pred_np[scores >= np.median(scores)] = 1
        acc = np.sum(pred_np == labels) / len(labels)
        logger.info('Epoch {} acc {:.4f} auc {:.4f}'.format(epoch, acc, auc))

    return auc


def save_model(tosave_model):
    model_path = 'ssl_model-' + str(args.image_size) + '-' + args.dataset + '.pth'
    save_path = os.path.join('./SSL_model', model_path)
    torch.save(tosave_model.module.state_dict(), save_path)


logger.info('Start training...')
logger.info("Random Seed is {}".format(args.manualSeed))
best_auc = 0
for epoch in range(args.max_epochs):
    logger.info('Epoch: %d start!' % epoch)

    epoch_start = time.time()

    train_epoch(epoch)
    test_auc = test_epoch(epoch)
    if test_auc > best_auc:
        best_auc = test_auc
        best_epoch = epoch
        save_model(SSL_model)

    logger.info('Epoch: {} finish with time {}!'.format(epoch, time.time()-epoch_start))

logger.info('Best auc {:.4f} on epoch {}'.format(best_auc, best_epoch))
