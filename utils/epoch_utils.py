import numpy as np
import torch
from torch.autograd import Variable
import time
import AverageMeter
from SRDataset import ImageDataset, SRDataset
import os
from resnet_roi import load_ssl_model

from utils import transforms
from utils.logger import logger


def get_union_box(bboxes):
    b,n = bboxes.shape[0], bboxes.shape[1]
    union_boxes = np.zeros((b, n, n, 4), dtype=np.float32)
    for i in range(b):
        for j in range(n):
            for k in range(n):
                union_boxes[i, j, k, 0] = min(bboxes[i, j, 0], bboxes[i, k, 0])
                union_boxes[i, j, k, 1] = min(bboxes[i, j, 1], bboxes[i, k, 1])
                union_boxes[i, j, k, 2] = max(bboxes[i, j, 2], bboxes[i, k, 2])
                union_boxes[i, j, k, 3] = max(bboxes[i, j, 3], bboxes[i, k, 3])
    union_boxes = torch.from_numpy(union_boxes)
    return union_boxes.view((b, -1, 4))


def generate_dataloader(model, eval_type, args):
    cache_size = 256
    if args.image_size == 448:
        cache_size = 256 * 2
    if args.image_size == 352:
        cache_size = 402

    transform_train = transforms.Compose([
        transforms.Resize((cache_size, cache_size)),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((cache_size, cache_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if eval_type == 'train':
        data = ImageDataset(max_person=args.max_person+1, image_dir=args.images_root,
                              images_list=args.train_file_pre + '_images.txt',
                              bboxes_list=args.train_file_pre + '_bbox.json',
                              image_size=args.image_size, input_transform=transform_train)
    elif eval_type == 'valid':
        data = ImageDataset(max_person=args.max_person + 1, image_dir=args.images_root, \
                                       images_list=args.valid_file_pre + '_images.txt', \
                                       bboxes_list=args.valid_file_pre + '_bbox.json', \
                                       image_size=args.image_size, \
                                       input_transform=transform_test)
    else:
        data = ImageDataset(max_person=args.max_person + 1, image_dir=args.images_root, \
                                      images_list=args.test_file_pre + '_images.txt', \
                                      bboxes_list=args.test_file_pre + '_bbox.json', \
                                      image_size=args.image_size, \
                                      input_transform=transform_test)
    loader = torch.utils.data.DataLoader(data, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=0,
                                             worker_init_fn=np.random.seed(args.manualSeed))

    model.eval()
    batch_time = AverageMeter.AverageMeter()
    end_time = time.time()

    union_filename = './para_data/' + args.dataset + '_' + eval_type + '_union' + '.npy'
    feat_filename = './para_data/' + args.dataset + '_' + eval_type + '_ssl.npy'
    ssl_model = load_ssl_model()
    ssl_model.eval()
    if args.cuda:
        ssl_model.cuda()
        ssl_model = torch.nn.DataParallel(ssl_model)

    if os.path.exists(feat_filename) and os.path.exists(union_filename) and not args.regenerate_roifeat:
        all_feat = np.load(feat_filename)
        logger.info('loadding RoI feature npy from {} successfully'.format(feat_filename))
        all_union_feat = np.load(union_filename, mmap_mode='r')  # [B, N, N, 2048]
        logger.info('loading union npy from {} successfully'.format(union_filename))
    else:
        all_feat, all_union_feat = [], []
        for batch_idx, (img, image_bboxes) in enumerate(loader):
            if args.cuda:
                img, image_bboxes = img.cuda(), image_bboxes.cuda()
            img, image_bboxes = Variable(img), Variable(image_bboxes)
            node_num = image_bboxes.shape[1]

            union_boxes = get_union_box(image_bboxes)
            if args.cuda:
                union_boxes = union_boxes.cuda()
            image_bboxes = torch.cat((image_bboxes, union_boxes), axis=1)
            del union_boxes
            image_bboxes = Variable(image_bboxes)

            # [batcn, node_num, 2048]
            rois_feature_all = model(img, image_bboxes)
            feature_num = rois_feature_all.shape[2]
            rois_feature = rois_feature_all[:, :node_num, :]
            union_feature = rois_feature_all[:, node_num:, :].reshape(-1, node_num, node_num, feature_num)
            if args.load_ssl_model:
                img_feat = ssl_model(img, image_bboxes)  # [batch_size, max_person+1, feat_dim]
                rois_feature = torch.cat((rois_feature, img_feat[:, -1:, :]), dim=1)

            all_feat.append(rois_feature.cpu().data.numpy())
            all_union_feat.append(union_feature.cpu().data.numpy())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % args.print_freq == 0:
                logger.info('%s Epoch: [%d/%d]  '
                            'Time %.3f (%.3f)\t'
                            % (eval_type, batch_idx, len(loader),
                               batch_time.val, batch_time.avg))

        all_feat = np.concatenate(all_feat)
        all_union_feat = np.concatenate(all_union_feat)
        np.save(feat_filename, all_feat)
        np.save(union_filename, all_union_feat)

    class_weight, class_count = [], []

    if eval_type == 'train':
        dataset = SRDataset(all_feat, all_union_feat, max_person=args.max_person+1, image_dir=args.images_root, \
            images_list=args.train_file_pre + '_images.txt',
            relations_list=args.train_file_pre + '_relation.json', image_size=args.image_size)
        class_weight, class_count = dataset.class_weight()
        batch_size = args.batch_size
        is_shuffle = True
    elif eval_type == 'valid':
        dataset = SRDataset(all_feat, all_union_feat, max_person=args.max_person+1,image_dir=args.images_root, \
            images_list=args.valid_file_pre + '_images.txt',
            relations_list=args.valid_file_pre + '_relation.json', image_size=args.image_size)
        batch_size = args.test_batch_size
        is_shuffle = False
    else:
        dataset = SRDataset(all_feat, all_union_feat, max_person=args.max_person+1,image_dir=args.images_root, \
                            images_list=args.test_file_pre + '_images.txt', \
                            relations_list=args.test_file_pre + '_relation.json', image_size=args.image_size
                            )
        batch_size = args.test_batch_size
        is_shuffle = False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle,
                                         num_workers=args.num_workers,
                                         worker_init_fn=np.random.seed(args.manualSeed))

    return dataloader, class_weight, class_count
