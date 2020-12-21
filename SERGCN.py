import torch.nn as nn

import RGCN

class SERGCN(nn.Module):
    def __init__(self, num_class=5, hidden_dim=2048, time_step=3, node_num=5):
        super(SERGCN, self).__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.time_step = time_step
        self.node_num = node_num

        self.rgcn = RGCN.RGCN(state_dim=hidden_dim, node_num=node_num,
                              edge_types=num_class, time_step=time_step
                              )
    
    # batch_bboxes are transformed with dim [batch, node_num, 4]
    # return with [batch, node_num, node_num, num_class]
    def forward(self, rois_feature, union_feat, full_mask):

        feat_body = rois_feature[:, :-2, :]
        feat_img_1 = union_feat[:, :-1, :-1, :]  # feature union
        feat_img_2 = rois_feature[:, -2, :]  # image feature I

        feat_img_ssl = rois_feature[:, -1, :]
        feat_img = [feat_img_1, feat_img_2]

        graph_scores = self.rgcn(feat_body, feat_img, feat_img_ssl, full_mask)
        return graph_scores


class SSLModel(nn.Module):
    def __init__(self, model):
        super(SSLModel, self).__init__()

        self.resnet = nn.Sequential(*list(model.children())[:-1])

        self.fc = nn.Bilinear(1024*2, 1024*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, pos_img, neg_img):
        feat_img = self.resnet(img)
        feat_pos_img = self.resnet(pos_img)
        feat_neg_img = self.resnet(neg_img)

        # feat_img = self.tanh(self.feat_map(feat_img.view(-1, 2048)))
        # feat_pos_img = self.tanh(self.feat_map(feat_pos_img.view(-1, 2048)))
        # feat_neg_img = self.tanh(self.feat_map(feat_neg_img.view(-1, 2048)))
        feat_img = feat_img.view(-1, 2048)
        feat_pos_img = feat_pos_img.view(-1, 2048)
        feat_neg_img = feat_neg_img.view(-1, 2048)

        score_pos = self.sigmoid(self.fc(feat_img, feat_pos_img))
        score_neg = self.sigmoid(self.fc(feat_img, feat_neg_img))

        return score_pos, score_neg
