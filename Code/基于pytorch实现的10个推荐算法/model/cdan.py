"""
@Author: 344733705@qq.com

@info:
2023.03.015: 加入取单塔功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.core import DNN
from model.cdan_tower import CdanTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim
from preprocessing.utils import Cosine_Similarity
from preprocessing.utils import mmd_rbf


class CDAN(CdanTower):
    """EDM模型"""

    def __init__(self,
                 user_columns,
                 item_columns,
                 popular_columns,
                 item_vocabulary_size,
                 alpha=0.5,
                 gamma=1,
                 dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128),
                 dnn_activation='relu',
                 l2_reg_dnn=0,
                 l2_reg_embedding=1e-6,
                 dnn_dropout=0,
                 init_std=0.0001,
                 seed=1024,
                 task='binary',
                 device='cpu',
                 gpus=None):
        super(CDAN, self).__init__(user_columns,
                                   item_columns,
                                   popular_columns,
                                   l2_reg_embedding=l2_reg_embedding,
                                   init_std=init_std,
                                   seed=seed,
                                   task=task,
                                   device=device,
                                   gpus=gpus)

        self.alpha = alpha
        self.item_embed_dict = nn.Embedding(item_vocabulary_size + 1,
                                            dnn_hidden_units[-1],
                                            padding_idx=0,
                                            sparse=False)

        if len(user_columns) > 0:
            self.user_feature_dnn = DNN(compute_input_dim(user_columns), dnn_hidden_units,
                                        activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                        use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.user_dnn_embedding = None

        if len(item_columns) > 0:
            self.item_pro_feature_dnn = DNN(compute_input_dim(item_columns), dnn_hidden_units,
                                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                            use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_hf_pro_dnn_embedding = None
            self.item_lt_pro_dnn_embedding = None
            self.item_pro_dnn_embedding = None
            self.item_pop_feature_dnn = DNN(compute_input_dim(item_columns), dnn_hidden_units,
                                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                            use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_hf_pop_dnn_embedding = None
            self.item_lt_pop_dnn_embedding = None
            self.item_pop_dnn_embedding = None

        if len(popular_columns) > 0:
            self.popular_feature_dnn = DNN(compute_input_dim(popular_columns), dnn_hidden_units,
                                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                           use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.popular_dnn_embedding = None

        hybrid_item_columns = popular_columns + item_columns
        if len(hybrid_item_columns) > 0:
            self.hybrid_item_feature_dnn = DNN(dnn_hidden_units[-1] * 2, dnn_hidden_units,
                                               activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                               use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.hybrid_item_dnn_embedding = None

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus

    def get_item_inner_loss(self, item_pro_embed, sim_item_ids):
        sim_center = self.item_embed_dict(sim_item_ids).mean(axis=1)
        item_inner_loss = Cosine_Similarity(item_pro_embed,
                                            sim_center,
                                            gamma=self.gamma).sum()
        return item_inner_loss

    def get_MMD_loss(self, item_pro_embed_hf, item_pro_embed_lf):
        MMD_loss = mmd_rbf(item_pro_embed_hf, item_pro_embed_lf)
        return MMD_loss

    def get_pop_sim_loss(self, popular_embed, item_pop_embed_hf):
        loss = (Cosine_Similarity(popular_embed,
                                  item_pop_embed_hf,
                                  gamma=self.gamma)).sum()
        return loss

    def get_orthogonal_loss(self, item_pop_embed_hf, item_pro_embed_hf):
        loss = (1 - Cosine_Similarity(item_pop_embed_hf,
                                      item_pro_embed_hf,
                                      gamma=self.gamma)).sum()
        return loss

    @staticmethod
    def get_unbiased_loss(item_pro_embed_hf, user_embedding, labels):
        labels[labels == 0] = -1
        sim_mat = torch.matmul(item_pro_embed_hf, user_embedding.T)
        batch_sim_mat = sim_mat.sum(axis=1)
        user_item_sim = torch.diagonal(sim_mat)
        ratio = user_item_sim / batch_sim_mat
        loss = (torch.where(torch.isnan(ratio), torch.full_like(ratio, 0), ratio) * labels).sum()
        if torch.isnan(loss).any():
            print("loss error!")
        return loss

    @staticmethod
    def get_biased_loss(hybrid_item_embedding, user_embedding, labels):
        labels[labels == 0] = -1
        sim_mat = torch.matmul(hybrid_item_embedding, user_embedding.T)
        batch_sim_mat = sim_mat.sum(axis=1)
        user_item_sim = torch.diagonal(sim_mat)
        ratio = user_item_sim / batch_sim_mat
        loss = (torch.where(torch.isnan(ratio), torch.full_like(ratio, 0), ratio) * labels).sum()
        if torch.isnan(loss).any():
            print("loss error!")
        return loss

    def forward1(self, inputs, labels, sim_items_list, is_train=True):
        lt_idxs = torch.where(inputs[:, -1].int() == 1)[0].long()
        hf_idxs = torch.where(inputs[:, -1].int() == 0)[0].long()
        item_ids = inputs[:, 0].int().numpy()
        #
        inputs = torch.concat([inputs[lt_idxs, 1:-1], inputs[hf_idxs, 1:-1]], axis=0)
        labels = torch.concat([labels[lt_idxs, :], labels[hf_idxs, :]], axis=0)
        inputs_lt = inputs[lt_idxs, 1:-1]
        inputs_hf = inputs[hf_idxs, 1:-1]

        item_pro_dnn_embedding = None
        item_dnn_embedding = None

        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_varlen_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs,
                                                self.user_dnn_feature_columns,
                                                self.user_embedding_dict)

            user_sparse_embedding_list += user_varlen_embedding_list
            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
            self.user_dnn_embedding = self.user_feature_dnn(user_dnn_input)

        if len(self.item_dnn_feature_columns) > 0:
            # high frequent items pro
            item_hf_pro_sparse_embedding_list, item_hf_varlen_embedding_list, item_hf_pro_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)
            item_hf_pro_sparse_embedding_list += item_hf_varlen_embedding_list
            item_hf_pro_dnn_input = combined_dnn_input(item_hf_pro_sparse_embedding_list, item_hf_pro_dense_value_list)
            self.item_pro_dnn_embedding = self.item_pro_feature_dnn(item_hf_pro_dnn_input)
            self.item_lt_pro_dnn_embedding = self.item_pro_dnn_embedding[lt_idxs]
            self.item_hf_pro_dnn_embedding = self.item_pro_dnn_embedding[hf_idxs]

            # high frequent items pop
            item_hf_pop_sparse_embedding_list, item_hf_varlen_embedding_list, item_hf_pop_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)
            item_hf_pop_sparse_embedding_list += item_hf_varlen_embedding_list
            item_hf_pop_dnn_input = combined_dnn_input(item_hf_pop_sparse_embedding_list, item_hf_pop_dense_value_list)
            self.item_pop_dnn_embedding = self.item_pop_feature_dnn(item_hf_pop_dnn_input)
            self.item_lt_pop_dnn_embedding = self.item_pop_dnn_embedding[lt_idxs]
            self.item_hf_pop_dnn_embedding = self.item_pop_dnn_embedding[hf_idxs]

            if is_train:
                self.item_embed_dict.weight.data[item_ids, :] = torch.FloatTensor(self.item_pro_dnn_embedding)

        if len(self.popular_dnn_feature_columns) > 0:
            # popularity
            popular_sparse_embedding_list, popular_varlen_embedding_list, popular_dense_value_list = \
                self.input_from_feature_columns(inputs, self.popular_dnn_feature_columns, self.popular_embedding_dict)
            popular_sparse_embedding_list += popular_varlen_embedding_list
            popular_dnn_input = combined_dnn_input(popular_sparse_embedding_list, popular_dense_value_list)
            self.popular_dnn_embedding = self.popular_feature_dnn(popular_dnn_input)

        if len(self.item_dnn_feature_columns) > 0 and len(self.popular_dnn_feature_columns) > 0:
            # v(x_i)
            hybrid_item_dnn_input = torch.concat([self.popular_dnn_embedding, self.item_pro_dnn_embedding],
                                                 axis=-1)
            self.hybrid_item_dnn_embedding = self.hybrid_item_feature_dnn(hybrid_item_dnn_input)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            item_sim_loss = self.get_item_inner_loss(self.item_pro_dnn_embedding,
                                                     sim_items_list)
            domain_sim_loss = self.get_MMD_loss(self.item_hf_pro_dnn_embedding,
                                                self.item_lt_pro_dnn_embedding)
            pop_sim_loss = self.get_pop_sim_loss(self.popular_dnn_embedding,
                                                 self.item_pop_dnn_embedding)
            pop_orthogonal_loss = self.get_orthogonal_loss(self.item_pop_dnn_embedding,
                                                           self.item_pro_dnn_embedding)
            unbiased_loss = self.get_unbiased_loss(self.item_pro_dnn_embedding,
                                                   self.user_dnn_embedding,
                                                   labels)
            biased_loss = self.get_biased_loss(self.hybrid_item_dnn_embedding,
                                               self.user_dnn_embedding,
                                               labels)
            loss = item_sim_loss + domain_sim_loss + pop_sim_loss + pop_orthogonal_loss + unbiased_loss + biased_loss

            item_embedding = torch.cat([self.alpha * self.item_pro_dnn_embedding,
                                        (1 - self.alpha) * self.hybrid_item_dnn_embedding], dim=-1)
            user_embedding = torch.cat([self.user_dnn_embedding, self.user_dnn_embedding], dim=-1)

            score = Cosine_Similarity(self.item_pro_dnn_embedding, self.user_dnn_embedding, gamma=self.gamma)
            output = self.out(score)
            if torch.isnan(output).any():
                print("nan!")
            loss_binary = F.binary_cross_entropy(output.float(), labels.squeeze().float(), reduction='sum')
            loss = loss_binary

            return loss, output

        elif len(self.user_dnn_feature_columns) > 0:
            user_embedding = torch.cat([self.user_dnn_embedding, self.user_dnn_embedding], dim=-1)
            return [], user_embedding
        elif len(self.item_dnn_feature_columns) > 0:
            item_embedding = torch.cat([self.alpha * self.item_pro_dnn_embedding,
                                        (1 - self.alpha) * self.hybrid_item_dnn_embedding], dim=-1)
            return [], item_embedding
        else:
            raise Exception("input Error! user and item feature columns are empty.")

    def forward2(self, inputs, labels, sim_items_list, is_train=True):
        lt_idxs = torch.where(inputs[:, -1].int() == 1)[0].long()
        hf_idxs = torch.where(inputs[:, -1].int() == 0)[0].long()
        item_ids = inputs[:, 0].int().numpy()
        #
        inputs = torch.concat([inputs[lt_idxs, 1:-1], inputs[hf_idxs, 1:-1]], axis=0)
        labels = torch.concat([labels[lt_idxs, :], labels[hf_idxs, :]], axis=0)
        inputs_lt = inputs[lt_idxs, 1:-1]
        inputs_hf = inputs[hf_idxs, 1:-1]

        item_pro_dnn_embedding = None
        item_dnn_embedding = None

        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_varlen_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs,
                                                self.user_dnn_feature_columns,
                                                self.user_embedding_dict)

            user_sparse_embedding_list += user_varlen_embedding_list
            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
            self.user_dnn_embedding = self.user_feature_dnn(user_dnn_input)

        if len(self.item_dnn_feature_columns) > 0:
            # high frequent items pro
            item_hf_pro_sparse_embedding_list, item_hf_varlen_embedding_list, item_hf_pro_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)
            item_hf_pro_sparse_embedding_list += item_hf_varlen_embedding_list
            item_hf_pro_dnn_input = combined_dnn_input(item_hf_pro_sparse_embedding_list, item_hf_pro_dense_value_list)
            self.item_pro_dnn_embedding = self.item_pro_feature_dnn(item_hf_pro_dnn_input)
            self.item_lt_pro_dnn_embedding = self.item_pro_dnn_embedding[lt_idxs]
            self.item_hf_pro_dnn_embedding = self.item_pro_dnn_embedding[hf_idxs]

            # high frequent items pop
            item_hf_pop_sparse_embedding_list, item_hf_varlen_embedding_list, item_hf_pop_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)
            item_hf_pop_sparse_embedding_list += item_hf_varlen_embedding_list
            item_hf_pop_dnn_input = combined_dnn_input(item_hf_pop_sparse_embedding_list, item_hf_pop_dense_value_list)
            self.item_pop_dnn_embedding = self.item_pop_feature_dnn(item_hf_pop_dnn_input)
            self.item_lt_pop_dnn_embedding = self.item_pop_dnn_embedding[lt_idxs]
            self.item_hf_pop_dnn_embedding = self.item_pop_dnn_embedding[hf_idxs]

            if is_train:
                self.item_embed_dict.weight.data[item_ids, :] = torch.FloatTensor(self.item_pro_dnn_embedding)

        if len(self.popular_dnn_feature_columns) > 0:
            # popularity
            popular_sparse_embedding_list, popular_varlen_embedding_list, popular_dense_value_list = \
                self.input_from_feature_columns(inputs, self.popular_dnn_feature_columns, self.popular_embedding_dict)
            popular_sparse_embedding_list += popular_varlen_embedding_list
            popular_dnn_input = combined_dnn_input(popular_sparse_embedding_list, popular_dense_value_list)
            self.popular_dnn_embedding = self.popular_feature_dnn(popular_dnn_input)

        if len(self.item_dnn_feature_columns) > 0 and len(self.popular_dnn_feature_columns) > 0:
            # v(x_i)
            hybrid_item_dnn_input = torch.concat([self.popular_dnn_embedding, self.item_pro_dnn_embedding],
                                                 axis=-1)
            self.hybrid_item_dnn_embedding = self.hybrid_item_feature_dnn(hybrid_item_dnn_input)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            item_sim_loss = self.get_item_inner_loss(self.item_pro_dnn_embedding,
                                                     sim_items_list)
            domain_sim_loss = self.get_MMD_loss(self.item_hf_pro_dnn_embedding,
                                                self.item_lt_pro_dnn_embedding)
            pop_sim_loss = self.get_pop_sim_loss(self.popular_dnn_embedding,
                                                 self.item_pop_dnn_embedding)
            pop_orthogonal_loss = self.get_orthogonal_loss(self.item_pop_dnn_embedding,
                                                           self.item_pro_dnn_embedding)
            unbiased_loss = self.get_unbiased_loss(self.item_pro_dnn_embedding,
                                                   self.user_dnn_embedding,
                                                   labels)
            biased_loss = self.get_biased_loss(self.hybrid_item_dnn_embedding,
                                               self.user_dnn_embedding,
                                               labels)
            loss = item_sim_loss + domain_sim_loss + pop_sim_loss + pop_orthogonal_loss + unbiased_loss + biased_loss

            item_embedding = torch.cat([self.alpha * self.item_pro_dnn_embedding,
                                        (1 - self.alpha) * self.hybrid_item_dnn_embedding], dim=-1)
            user_embedding = torch.cat([self.user_dnn_embedding, self.user_dnn_embedding], dim=-1)

            score = Cosine_Similarity(self.item_pro_dnn_embedding, self.user_dnn_embedding, gamma=self.gamma)
            output = self.out(score)
            if torch.isnan(output).any():
                print("nan!")
            loss_binary = F.binary_cross_entropy(output.float(), labels.squeeze().float(), reduction='sum')
            loss = loss_binary

            return loss, output

        elif len(self.user_dnn_feature_columns) > 0:
            user_embedding = torch.cat([self.user_dnn_embedding, self.user_dnn_embedding], dim=-1)
            return [], user_embedding
        elif len(self.item_dnn_feature_columns) > 0:
            item_embedding = torch.cat([self.alpha * self.item_pro_dnn_embedding,
                                        (1 - self.alpha) * self.hybrid_item_dnn_embedding], dim=-1)
            return [], item_embedding
        else:
            raise Exception("input Error! user and item feature columns are empty.")

    def forward(self, inputs,
                labels,
                sim_items_list=None,
                item_ids=None,
                lt_flags=None,
                is_train=True):
        # inputs = inputs[:, 1:-1]
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_varlen_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs,
                                                self.user_dnn_feature_columns,
                                                self.user_embedding_dict)

            user_sparse_embedding_list += user_varlen_embedding_list
            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
            self.user_dnn_embedding = self.user_feature_dnn(user_dnn_input)

        if len(self.item_dnn_feature_columns) > 0:
            # high frequent items pro
            item_hf_pro_sparse_embedding_list, item_hf_varlen_embedding_list, item_hf_pro_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)
            item_hf_pro_sparse_embedding_list += item_hf_varlen_embedding_list
            item_hf_pro_dnn_input = combined_dnn_input(item_hf_pro_sparse_embedding_list, item_hf_pro_dense_value_list)
            self.item_pro_dnn_embedding = self.item_pro_feature_dnn(item_hf_pro_dnn_input)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            score = Cosine_Similarity(self.item_pro_dnn_embedding, self.user_dnn_embedding, gamma=self.gamma)
            output = self.out(score)
            if torch.isnan(output).any():
                print("nan!")
            loss = F.binary_cross_entropy(output.float(), labels.squeeze().float(), reduction='sum')
            return loss, output

        elif len(self.user_dnn_feature_columns) > 0:
            user_embedding = torch.cat([self.user_dnn_embedding, self.user_dnn_embedding], dim=-1)
            return [], user_embedding
        elif len(self.item_dnn_feature_columns) > 0:
            item_embedding = torch.cat([self.alpha * self.item_pro_dnn_embedding,
                                        (1 - self.alpha) * self.hybrid_item_dnn_embedding], dim=-1)
            return [], item_embedding
        else:
            raise Exception("input Error! user and item feature columns are empty.")
