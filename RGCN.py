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
        self.node_fc_u = nn.ModuleList()
        self.node_fc_v = nn.ModuleList()
        self.edge_fc = nn.ModuleList()
        self.edge_prob = nn.ModuleList()
        for t in range(self.time_step):
            self.node_fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.node_fc_u.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.node_fc_v.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.edge_fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.edge_prob.append(nn.Linear(self.hidden_dim, 1))
        # self.img_fc = nn.Linear(self.state_dim, self.hidden_dim)

        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        # self.scene_emb = nn.Embedding(365, 512)

        edge_fc_input = self.hidden_dim * 3

        self.mlp = nn.Sequential(
            nn.Linear(edge_fc_input, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.edge_types)
        )
        self.img_fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.img_fc2 = nn.Linear(self.state_dim, self.hidden_dim)

        self._initialization()

    # feat_body with feature dim [batch, node_num, hidden_state_dim]
    # feat_img [batch, hidden_state_dim]
    # return output with feature dim [batch, node_num, output_dim]
    def forward(self, feat_body, feat_img, feat_img_ssl, full_mask):

        [feat_img_1, feat_img_2] = feat_img  # [feature union, image feature]

        full_mask = full_mask.view(-1, self.node_num, self.node_num, 1).detach()
        full_mask_edge = full_mask.repeat(1, 1, 1,
                                          self.edge_types).float().detach()  # [batch_size, node_num, node_num, num_classes]
        full_mask_node = full_mask.repeat(1, 1, 1, self.hidden_dim).float().detach()

        node_prop_state = feat_body
        edge_prop_state = feat_img_1

        all_feat_edge = []

        for t in range(self.time_step):
            node_message_states = self.node_fc[t](node_prop_state)  # node feature, [batch, node_num, hidden_dim]
            node_message_states_u = self.node_fc_u[t](node_prop_state)
            node_message_states_v = self.node_fc_v[t](node_prop_state)
            edge_message_states = self.edge_fc[t](edge_prop_state)

            feature_row_large = node_message_states.contiguous().view(-1, self.node_num, 1, self.hidden_dim).repeat(1, 1,
                                                                                                               self.node_num,
                                                                                                               1)
            feature_col_large = node_message_states.contiguous().view(-1, 1, self.node_num, self.hidden_dim).repeat(1,
                                                                                                               self.node_num,
                                                                                                               1, 1)
            feature_col_large_v = node_message_states_v.contiguous().view(-1, 1, self.node_num, self.hidden_dim).repeat(1,
                                                                                                               self.node_num,
                                                                                                               1, 1)
            edge_feat = feature_row_large + feature_col_large + edge_message_states
            #edge_prop_state = edge_prop_state + self.act(self.bn2(edge_feat.permute(0,3,1,2)).permute(0,2,3,1))  # edge embedding, [batch, node_num, node_num, hidden_dim]
            edge_prop_state = edge_prop_state + self.act(edge_feat)  # edge embedding, [batch, node_num, node_num, hidden_dim]

            all_feat_edge.append(edge_prop_state)
            # softmax
            # edge_prob = self.edge_prob[t](edge_prop_state) #* full_mask + (1-full_mask) * (-1e6) # [batch, node_num, node_num, 1]
            # edge_prob = torch.softmax(edge_prob, dim=-2) * full_mask
            # sigmoid
            edge_prob = self.sigmoid(edge_prop_state)
            edge_prob = edge_prob / (torch.sum(edge_prob * full_mask, dim=-2, keepdim=True) + 1e-6) * full_mask
            # FC
            # edge_prob = self.edge_prob[t](edge_prop_state)
            # edge_prob = edge_prop_state
            node_feat = (node_message_states_u + torch.sum(edge_prob * feature_col_large_v * full_mask_node,dim=-2)) / (torch.sum(full_mask, dim=-2) + 1)
            #node_prop_state = node_prop_state + self.act(self.bn1(node_feat.permute(0,2,1)).permute(0,2,1))  # [batch_size, node_num, hidden_dim]
            node_prop_state = node_prop_state + self.act(node_feat)  # [batch_size, node_num, hidden_dim]

        max_pool, _ = torch.max(torch.stack(all_feat_edge), dim=0)

        # feature_img_1 = self.img_fc(feat_img_1)
        # feature_img_2 = feat_img_2.contiguous().\
        #     view(-1, 1, 1, self.hidden_dim).repeat(1, self.node_num,self.node_num, 1)
        #
        # feature_img_ssl = feat_img_ssl.contiguous().\
        #     view(-1, 1, 1, self.hidden_dim).repeat(1, self.node_num, self.node_num, 1)
        feature_img_2 = self.img_fc1(feat_img_2).contiguous(). \
            view(-1, 1, 1, self.hidden_dim).repeat(1, self.node_num, self.node_num, 1)

        feature_img_ssl = self.img_fc2(feat_img_ssl).contiguous(). \
            view(-1, 1, 1, self.hidden_dim).repeat(1, self.node_num, self.node_num, 1)

        feature_all = torch.cat((max_pool, feature_img_2, feature_img_ssl), dim=3)
        relation_score = self.mlp(feature_all).contiguous()  # [batch_size, node_num, node_num, num_classes]
        graph_scores = relation_score * full_mask_edge

        return graph_scores

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)
