"""
@Author: 344733705@qq.com

@info:
2023.03.015: 加入取单塔功能
"""

import math

import torch
import torch.nn.functional as F

from layers.core import DNN
from model.dccl_tower import DcclTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim
from preprocessing.utils import Cosine_Similarity
from preprocessing.utils import mmd_rbf


class DCCL(DcclTower):
    """EDM模型"""

    def __init__(self,
                 user_columns,
                 item_columns,
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
        super(DCCL, self).__init__(user_columns,
                                   item_columns,
                                   l2_reg_embedding=l2_reg_embedding,
                                   init_std=init_std,
                                   seed=seed,
                                   task=task,
                                   device=device,
                                   gpus=gpus)

        if len(user_columns) > 0:
            self.user_pro_feature_dnn = DNN(compute_input_dim(user_columns), dnn_hidden_units,
                                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                            use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.user_pop_feature_dnn = DNN(compute_input_dim(user_columns), dnn_hidden_units,
                                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                            use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.user_pro_dnn_embedding = None
            self.user_pop_dnn_embedding = None
            self.user_dnn_embedding = None

        if len(item_columns) > 0:
            self.item_pro_feature_dnn = DNN(compute_input_dim(item_columns), dnn_hidden_units,
                                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                            use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_pop_feature_dnn = DNN(compute_input_dim(item_columns), dnn_hidden_units,
                                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                            use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_pro_dnn_embedding = None
            self.item_pop_dnn_embedding = None
            self.item_dnn_embedding = None

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

    @staticmethod
    def get_mmd_loss(item_pro_embed_hf, item_pro_embed_lf):
        MMD_loss = mmd_rbf(item_pro_embed_hf, item_pro_embed_lf)
        return MMD_loss

    def get_pop_sim_loss(self, popular_embed, item_pop_embed_hf):
        loss = (Cosine_Similarity(popular_embed,
                                  item_pop_embed_hf,
                                  gamma=self.gamma)).sum()
        return loss

    def get_orthogonal_loss(self, item_pop_embed_hf, item_pro_embed_hf):
        loss = (1 - abs(Cosine_Similarity(item_pop_embed_hf,
                                          item_pro_embed_hf,
                                          gamma=self.gamma))).sum()
        return loss

    @staticmethod
    def get_unbiased_loss(item_pro_embed_hf, user_embedding, labels):
        indicators = torch.tensor(labels)
        indicators[indicators == 0] = -1
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
        indicators = torch.tensor(labels)
        indicators[indicators == 0] = -1
        sim_mat = torch.matmul(hybrid_item_embedding, user_embedding.T)
        batch_sim_mat = sim_mat.sum(axis=1)
        user_item_sim = torch.diagonal(sim_mat)
        ratio = user_item_sim / batch_sim_mat
        loss = (torch.where(torch.isnan(ratio), torch.full_like(ratio, 0), ratio) * labels).sum()
        if torch.isnan(loss).any():
            print("loss error!")
        return loss

    @staticmethod
    def get_pro_loss(item_embed, user_embed, labels, pop_ratio):
        def func(x):
            if x == 0:
                return 0
            else:
                return math.exp(x)

        # N = item_embed.shape[0]
        ratios = torch.tensor(list(map(func, (-pop_ratio).tolist())), dtype=torch.double)
        pos_sim = torch.tensor(list(map(func, (Cosine_Similarity(item_embed,
                                                                 user_embed) * labels.squeeze()).tolist())),
                               dtype=torch.double)
        total_sim = torch.tensor(list(map(func, (Cosine_Similarity(item_embed,
                                                                   user_embed)).tolist())),
                                 dtype=torch.double)
        loss = math.log((ratios * pos_sim).sum() / total_sim.sum())

        return -loss

    @staticmethod
    def get_pop_loss(item_embed, user_embed, labels, pop_ratio):
        def func1(x):
            if x == 0:
                return 0
            else:
                return 1 - math.exp(x)

        def func2(x):
            if x == 0:
                return 0
            else:
                return math.exp(x)

        # N = item_embed.shape[0]
        ratios = torch.tensor(list(map(func1, (-pop_ratio).tolist())), dtype=torch.double)
        pos_sim = torch.tensor(list(map(func2, (Cosine_Similarity(item_embed,
                                                                  user_embed) * labels.squeeze()).tolist())),
                               dtype=torch.double)
        total_sim = torch.tensor(list(map(func2, (Cosine_Similarity(item_embed,
                                                                    user_embed)).tolist())),
                                 dtype=torch.double)
        loss = math.log((ratios * pos_sim).sum() / total_sim.sum())

        return -loss

    def forward(self,
                inputs,
                labels,
                pop_ratio
                ):

        if len(self.user_dnn_feature_columns) > 0:
            # users pro
            user_pro_sparse_embedding_list, \
            user_pro_varlen_embedding_list, \
            user_pro_dense_value_list = \
                self.input_from_feature_columns(inputs,
                                                self.user_dnn_feature_columns,
                                                self.user_pro_embedding_dict)
            user_pro_sparse_embedding_list += user_pro_varlen_embedding_list
            user_pro_dnn_input = combined_dnn_input(user_pro_sparse_embedding_list, user_pro_dense_value_list)
            self.user_pro_dnn_embedding = self.user_pro_feature_dnn(user_pro_dnn_input)

            # users pop
            user_pop_sparse_embedding_list, \
            user_pop_varlen_embedding_list, \
            user_pop_dense_value_list = \
                self.input_from_feature_columns(inputs,
                                                self.user_dnn_feature_columns,
                                                self.user_pop_embedding_dict)
            user_pop_sparse_embedding_list += user_pop_varlen_embedding_list
            user_pop_dnn_input = combined_dnn_input(user_pop_sparse_embedding_list, user_pop_dense_value_list)
            self.user_pop_dnn_embedding = self.user_pop_feature_dnn(user_pop_dnn_input)

            # pro + pop
            self.user_dnn_embedding = torch.concat([self.user_pro_dnn_embedding,
                                                    self.user_pop_dnn_embedding], axis=1)

        if len(self.item_dnn_feature_columns) > 0:
            # items pro
            item_pro_sparse_embedding_list, \
            item_pro_varlen_embedding_list, \
            item_pro_dense_value_list = \
                self.input_from_feature_columns(inputs,
                                                self.item_dnn_feature_columns,
                                                self.item_pro_embedding_dict)
            item_pro_sparse_embedding_list += item_pro_varlen_embedding_list
            item_pro_dnn_input = combined_dnn_input(item_pro_sparse_embedding_list, item_pro_dense_value_list)
            self.item_pro_dnn_embedding = self.item_pro_feature_dnn(item_pro_dnn_input)

            # items pop
            item_pop_sparse_embedding_list, \
            item_pop_varlen_embedding_list, \
            item_pop_dense_value_list = \
                self.input_from_feature_columns(inputs,
                                                self.item_dnn_feature_columns,
                                                self.item_pop_embedding_dict)
            item_pop_sparse_embedding_list += item_pop_varlen_embedding_list
            item_pop_dnn_input = combined_dnn_input(item_pop_sparse_embedding_list, item_pop_dense_value_list)
            self.item_pop_dnn_embedding = self.item_pop_feature_dnn(item_pop_dnn_input)

            # pro + pop
            self.item_dnn_embedding = torch.concat([self.item_pro_dnn_embedding,
                                                    self.item_pop_dnn_embedding], axis=1)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            loss_pro = self.get_pro_loss(self.item_pro_dnn_embedding,
                                         self.user_pro_dnn_embedding,
                                         labels,
                                         pop_ratio)
            loss_pop = self.get_pop_loss(self.item_pop_dnn_embedding,
                                         self.user_pop_dnn_embedding,
                                         labels,
                                         pop_ratio)

            score = Cosine_Similarity(self.item_dnn_embedding,
                                      self.user_dnn_embedding,
                                      gamma=self.gamma)
            output = self.out(score)
            if torch.isnan(output).any():
                print("nan!")
            loss_main = F.binary_cross_entropy(output.float(), labels.squeeze().float(), reduction='sum')

            loss_total = loss_main + 100*loss_pro + 100*loss_pop
            if float('nan') == loss_pro or float('nan') == loss_pop:
                print("nan!")
            if torch.isnan(torch.tensor(loss_total).any()):
                print("nan!")
            return loss_total, output

        elif len(self.user_dnn_feature_columns) > 0:

            return [], self.user_dnn_embedding
        elif len(self.item_dnn_feature_columns) > 0:

            return [], self.item_dnn_embedding
        else:
            raise Exception("input Error! user and item feature columns are empty.")
