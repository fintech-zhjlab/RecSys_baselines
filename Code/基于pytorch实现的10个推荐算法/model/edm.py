"""
@Author: 344733705@qq.com

@info:
2023.03.015: 加入取单塔功能
"""

from layers.core import DNN
from model.edm_tower import EdmTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim
from preprocessing.utils import Cosine_Similarity


class EDM(EdmTower):
    """EDM模型"""

    def __init__(self,
                 user_atom_columns,
                 user_side_columns,
                 item_atom_columns,
                 item_side_columns,
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
        super(EDM, self).__init__(user_atom_columns,
                                  user_side_columns,
                                  item_atom_columns,
                                  item_side_columns,
                                  l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std,
                                  seed=seed,
                                  task=task,
                                  device=device,
                                  gpus=gpus)

        if len(user_atom_columns) > 0:
            self.user_atom_dnn = DNN(compute_input_dim(user_atom_columns), dnn_hidden_units,
                                     activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                     use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.user_atom_dnn_embedding = None

        if len(user_side_columns) > 0:
            self.user_side_dnn = DNN(compute_input_dim(user_side_columns), dnn_hidden_units,
                                     activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                     use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.user_side_dnn_embedding = None

        if len(item_atom_columns) > 0:
            self.item_atom_dnn = DNN(compute_input_dim(item_atom_columns), dnn_hidden_units,
                                     activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                     use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_atom_dnn_embedding = None

        if len(item_side_columns) > 0:
            self.item_side_dnn = DNN(compute_input_dim(item_side_columns), dnn_hidden_units,
                                     activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                     use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_side_dnn_embedding = None

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus

    def forward(self, inputs):
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)

            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
            self.user_dnn_embedding = self.user_dnn(user_dnn_input)

        if len(self.item_dnn_feature_columns) > 0:
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

            item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
            self.item_dnn_embedding = self.item_dnn(item_dnn_input)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            score = Cosine_Similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
            output = self.out(score)
            return output

        elif len(self.user_dnn_feature_columns) > 0:
            return self.user_dnn_embedding

        elif len(self.item_dnn_feature_columns) > 0:
            return self.item_dnn_embedding

        else:
            raise Exception("input Error! user and item feature columns are empty.")
