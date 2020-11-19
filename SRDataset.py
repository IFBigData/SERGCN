import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
from collections import defaultdict
import random


class SRDataset(data.Dataset):
    def __init__(self, feat, union_feat, max_person, image_dir, images_list, relations_list, image_size):
        super(SRDataset, self).__init__()
        self.max_person = max_person
        self.image_dir = image_dir
        self.image_size = image_size
        self.names = []

        with open(images_list, 'r') as fin:
            for line in fin:
                self.names.append(line.split()[0])

        with open(relations_list, 'r') as fin:
            self.relations = json.load(fin)

        if 'PIPA' in self.image_dir:
            scene_file = '../scene_PIPA.txt'
        elif 'PISC' in self.image_dir:
            scene_file = '../scene_PISC.txt'
        else:
            raise FileNotFoundError

        self.scene_dict = defaultdict(list)
        with open(scene_file, 'r') as f:
            for line in f:
                tmp = line.split()
                self.scene_dict[tmp[0]].append(tmp[1])

        self.feat = torch.from_numpy(feat)
        self.union_feat = torch.from_numpy(union_feat)
        self.labels = []
        for image_relations in self.relations:
            img_labels = [image_relation[2] for image_relation in image_relations]
            self.labels.extend(img_labels)

    def __getitem__(self, index):

        relation_mask = np.zeros((self.max_person, self.max_person), dtype=np.int32)
        full_mask = np.zeros((self.max_person, self.max_person), dtype=np.int32)
        relation_id = np.zeros((self.max_person, self.max_person), dtype=np.int32)
        image_relations = self.relations[index]
        for i in range(len(image_relations)):
            image_relation = image_relations[i]
            relation_mask[image_relation[0]][image_relation[1]] = 1
            full_mask[image_relation[0]][image_relation[1]] = 1
            full_mask[image_relation[1]][image_relation[0]] = 1

            relation_id[image_relation[0]][image_relation[1]] = image_relation[2]
            relation_id[image_relation[1]][image_relation[0]] = image_relation[2]

        full_mask = torch.from_numpy(full_mask).long()
        relation_mask = torch.from_numpy(relation_mask).long()
        relation_id = torch.from_numpy(relation_id).long()

        # img: [3, image_size, image_size]
        # bbox_num: single number for the bbox number in this image
        # image_bboxes: [max_person, 4], the first bbox_num has real value, others are zero
        # relation_mask: [max_person, max_person] mask
        # relation_id: [max_person, max_person] labels
        return self.feat[index], self.union_feat[index], relation_mask[:-1, :-1], relation_id[:-1, :-1], full_mask[:-1, :-1]

    def __len__(self):
        return len(self.names)

    def class_weight(self):
        np_labels = np.array(self.labels)
        class_num = len(set(self.labels))
        class_sample_count = np.array([len(np.where(np_labels == t)[0]) for t in range(class_num)])
        weight = 1. / (class_sample_count) * len(self.labels)

        # weight = torch.from_numpy(weight)

        return weight, class_sample_count



class ImageDataset(data.Dataset):
    def __init__(self, max_person, image_dir, images_list, bboxes_list, image_size, input_transform=None):
        super(ImageDataset, self).__init__()
        self.max_person = max_person
        self.image_dir = image_dir
        self.image_size = image_size
        self.input_transform = input_transform
        self.names = []

        with open(images_list, 'r') as fin:
            for line in fin:
                self.names.append(line.split()[0])

        with open(bboxes_list, 'r') as fin:
            self.bboxes = json.load(fin)  # list of bboxes for each image [[bbox1_img1, bbox2_img1], [bbox1_img2]]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_dir, self.names[index])).convert('RGB')  # convert gray to rgb
        (w, h) = img.size

        bbox_num = len(self.bboxes[index])
        image_bboxes = np.zeros((self.max_person, 4), dtype=np.float32)
        bbox_np = np.array(self.bboxes[index])

        image_bboxes[:, 0] = 0
        image_bboxes[:, 1] = 0
        image_bboxes[:, 2] = w - 1
        image_bboxes[:, 3] = h - 1

        image_bboxes[0:bbox_num, :] = bbox_np[:, :]
        image_bboxes = torch.from_numpy(image_bboxes)

        if self.input_transform:
            img, image_bboxes = self.input_transform(img, image_bboxes)

        # img: [3, image_size, image_size]
        # image_bboxes: [max_person, 4], the first bbox_num has real value, others are zero
        return img, image_bboxes

    def __len__(self):
        return len(self.names)


# for ssl learning
class SceneDataset(data.Dataset):
    def __init__(self, image_dir, images_list_path, img_pos_dict_path, img_neg_dict_path, idxs, input_transform=None):
        super(SceneDataset, self).__init__()

        self.input_transform = input_transform
        self.image_dir = image_dir

        self.all_names = []
        with open(images_list_path, 'r') as fin:
            for line in fin:
                self.all_names.append(line.split()[0])
        self.names = [self.all_names[idx] for idx in idxs]

        with open(img_pos_dict_path, 'r') as fin:
            img_pos_dict_tmp = json.load(fin)
        with open(img_neg_dict_path, 'r') as fin:
            img_neg_dict_tmp = json.load(fin)
        self.img_pos_dict = {}
        self.img_neg_dict = {}
        for name in self.names:
            self.img_pos_dict[name] = img_pos_dict_tmp[name]
            self.img_neg_dict[name] = img_neg_dict_tmp[name]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_dir, self.names[index])).convert('RGB')

        name = self.names[index]
        pos_img_name = self.all_names[random.choice(self.img_pos_dict[name])]
        neg_img_name = self.all_names[random.choice(self.img_neg_dict[name])]
        # pos_img_name = self.all_names[self.img_pos_dict[name][random.randint(0,len(self.img_pos_dict[name])-1)]]
        # neg_img_name = self.all_names[self.img_neg_dict[name][random.randint(0,len(self.img_pos_dict[name])-1)]]
        pos_img = Image.open(os.path.join(self.image_dir, pos_img_name)).convert('RGB')
        neg_img = Image.open(os.path.join(self.image_dir, neg_img_name)).convert('RGB')

        if self.input_transform:
            img = self.input_transform(img)
            pos_img = self.input_transform(pos_img)
            neg_img = self.input_transform(neg_img)

        return img, pos_img, neg_img

    def __len__(self):
        return len(self.names)