import torch
import torch.nn as nn


class RGCN(nn.Module):
    def __init__(self, state_dim=10, node_num=5, edge_types=5, time_step=5):
        super(RGCN, self).__init__()
        self.time_step = time_step
        self.state_dim = state_dim  # 2048
        self.hidden_dim = state_dim
        self.edge_types = edge_types  # num of classes
        self.node_num = node_num

        # incoming and outgoing edge embedding
        self.node_fc = nn.ModuleList()
        for t in range(self.time_step):
            self.node_fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.img_fc = nn.Linear(self.state_dim, self.hidden_dim)

        self.act = nn.ReLU()
        self.scene_emb = nn.Embedding(365, 512)

        edge_fc_input = self.hidden_dim * 4

        self.edge_fc = nn.Sequential(
            nn.Linear(edge_fc_input, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.edge_types)
        )

        self._initialization()

    # feat_body with feature dim [batch, node_num, hidden_state_dim]
    # feat_img [batch, hidden_state_dim]
    # return output with feature dim [batch, node_num, output_dim]
    def forward(self, feat_body, feat_img, feat_img_ssl, full_mask):

        [feat_img_1, feat_img_2] = feat_img

        full_mask = full_mask.view(-1, self.node_num, self.node_num, 1).detach()
        full_mask_edge = full_mask.repeat(1, 1, 1,
                                          self.edge_types).float().detach()  # [batch_size, node_num, node_num, num_classes]
        full_mask_node = full_mask.repeat(1, 1, 1, self.hidden_dim).float().detach()

        prop_state = self.img_fc(feat_body)

        all_feat_edge = []

        for t in range(self.time_step):
            message_states = self.node_fc[t](prop_state)  # node feature, [batch, node_num, hidden_dim]

            feature_row_large = message_states.contiguous().view(-1, self.node_num, 1, self.hidden_dim).repeat(1, 1,
                                                                                                               self.node_num,
                                                                                                               1)
            feature_col_large = message_states.contiguous().view(-1, 1, self.node_num, self.hidden_dim).repeat(1,
                                                                                                               self.node_num,
                                                                                                               1, 1)
            if t == self.time_step:
                feature_edge = feature_row_large + feature_col_large
            else:
                feature_edge = self.act(
                    feature_row_large + feature_col_large)  # edge embedding, [batch, node_num, node_num, hidden_dim]
            all_feat_edge.append(feature_edge)
            prop_state = prop_state + self.act(
                message_states + torch.sum(feature_edge * feature_col_large * full_mask_node,
                                           dim=-2))  # [batch_size, node_num, state_dim//2]

        max_pool, _ = torch.max(torch.stack(all_feat_edge), dim=0)

        feature_img_1 = self.img_fc(feat_img_1)
        feature_img_2 = self.img_fc(feat_img_2).contiguous().\
            view(-1, 1, 1, self.hidden_dim).repeat(1, self.node_num,self.node_num, 1)

        feature_img_ssl = self.img_fc(feat_img_ssl).contiguous().\
            view(-1, 1, 1, self.hidden_dim).repeat(1, self.node_num, self.node_num, 1)

        feature_all = torch.cat((max_pool, feature_img_1, feature_img_2, feature_img_ssl), dim=3)
        relation_score = self.edge_fc(feature_all).contiguous()  # [batch_size, node_num, node_num, num_classes]
        graph_scores = relation_score * full_mask_edge

        return graph_scores

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)
