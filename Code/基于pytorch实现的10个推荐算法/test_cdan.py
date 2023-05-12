from collections import Counter

import numpy as np
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from model.cdan import CDAN
from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
import warnings

warnings.filterwarnings('ignore')

def data_process(data_path, samp_rows=10000):
    data = pd.read_csv(data_path, nrows=samp_rows)
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data = data.sort_values(by='timestamp', ascending=True)
    data = add_popular_features(data)
    train = data.iloc[:int(len(data) * 0.8)].copy()
    test = data.iloc[int(len(data) * 0.8):].copy()
    return train, test, data


def get_user_feature(data):
    data_group = data[data['rating'] == 1]
    data_group = data_group[['user_id', 'movie_id']].groupby('user_id').agg(list).reset_index()
    data_group['user_hist'] = data_group['movie_id'].apply(lambda x: '|'.join([str(i) for i in x]))
    data = pd.merge(data_group.drop('movie_id', axis=1), data, on='user_id')
    data_group = data[['user_id', 'rating']].groupby('user_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'user_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='user_id')
    return data


def get_item_feature(data):
    data_group = data[['movie_id', 'rating']].groupby('movie_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'item_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='movie_id')
    return data


def get_var_feature(data, col):
    key2index = {}

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",\
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    var_feature = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature)))
    max_len = max(var_feature_length)
    var_feature = pad_sequences(var_feature, maxlen=max_len, padding='post', )
    return key2index, var_feature, max_len


def get_test_var_feature(data, col, key2index, max_len):
    print("user_hist_list: \n")

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding='post')
    return test_hist


def get_longtail_dict(data, percentage=0.2):
    user_ids = data["user_id"]
    users_counter = Counter(user_ids)
    user_num = len(users_counter)
    hf_dict = dict(users_counter.most_common(int(user_num * percentage)))
    return hf_dict


def get_longtail_inputs(data, hf_dict):
    longtail_inputs = [0 if item in hf_dict else 1 for item in data["user_id"]]
    return longtail_inputs


def add_popular_features(data):
    movie_ids = data["movie_id"]
    movies_counter = Counter(movie_ids)
    movie_num = len(movies_counter)
    item_pop_dict = dict(movies_counter.most_common(movie_num))
    pop_features = [item_pop_dict[movie] for movie in movie_ids]
    data["pop_feat"] = pop_features
    return data


if __name__ == '__main__':
    # %%
    data_path = './data/movielens.txt'
    train, test, data = data_process(data_path, samp_rows=10000)
    train = get_user_feature(train)
    train = get_item_feature(train)

    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation']
    dense_features = ['user_mean_rating', 'item_mean_rating', 'pop_feat']

    # longtail_features = ['is_lt']
    target = ['rating']

    user_sparse_features, user_dense_features = ['user_id', 'gender', 'age', 'occupation'], ['user_mean_rating']
    item_sparse_features, item_dense_features = ['movie_id', ], ['item_mean_rating']
    popular_dense_features = ['pop_feat']

    # 注释：1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        train[feat] = lbe.transform(train[feat])
        test[feat] = lbe.transform(test[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(train[dense_features])
    train[dense_features] = mms.transform(train[dense_features])

    # 注释：2.preprocess the sequence feature
    genres_key2index, train_genres_list, genres_maxlen = get_var_feature(train, 'genres')
    user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')

    # 新增特征
    train_lt_dict = get_longtail_dict(train)
    train_lt_labels = get_longtail_inputs(train, train_lt_dict)
    train_item_ids = np.array(train["movie_id"])
    train_sim_item_groups = train_user_hist
    item_num = data["movie_id"].nunique()

    user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                            for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               user_dense_features]
    item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                            for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               item_dense_features]

    popular_feature_columns = [DenseFeat(feat, 1, ) for feat in popular_dense_features]

    item_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=1000, embedding_dim=4),
                                                    maxlen=genres_maxlen, combiner='mean', length_name=None)]

    user_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('user_hist', vocabulary_size=3470, embedding_dim=4),
                                                    maxlen=user_maxlen, combiner='mean', length_name=None)]

    # 注释：3.generate input data for model
    user_feature_columns += user_varlen_feature_columns
    item_feature_columns += item_varlen_feature_columns

    # 注释： add user history as user_varlen_feature_columns
    train_model_input = {name: train[name] for name in sparse_features + dense_features}
    train_model_input["genres"] = train_genres_list
    train_model_input["user_hist"] = train_user_hist

    # %%
    # 注释：4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = CDAN(user_feature_columns,
                 item_feature_columns,
                 popular_feature_columns,
                 item_num,
                 task='binary',
                 device=device)

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy'])

    # %%
    model.fit(train_model_input,
              train[target].values,
              train_lt_labels,
              train_item_ids,
              train_sim_item_groups,
              batch_size=256,
              epochs=10,
              verbose=2,
              validation_split=0.2)
    # model.save

    # %%
    # 注释：5.preprocess the test data
    test = pd.merge(test, train[['movie_id', 'item_mean_rating']].drop_duplicates(), on='movie_id', how='left').fillna(
        0.5)
    test = pd.merge(test, train[['user_id', 'user_mean_rating']].drop_duplicates(), on='user_id', how='left').fillna(
        0.5)
    test = pd.merge(test, train[['user_id', 'user_hist']].drop_duplicates(), on='user_id', how='left').fillna('1')
    test[dense_features] = mms.transform(test[dense_features])

    test_genres_list = get_test_var_feature(test, 'genres', genres_key2index, genres_maxlen)
    test_user_hist = get_test_var_feature(test, 'user_hist', user_key2index, user_maxlen)

    test_model_input = {name: test[name] for name in sparse_features + dense_features}
    test_model_input["genres"] = test_genres_list
    test_model_input["user_hist"] = test_user_hist

    # %%
    # 注释：6.Evaluate
    eval_tr = model.evaluate(train_model_input,
                             train[target].values,
                             train_lt_labels,
                             train_item_ids,
                             train_sim_item_groups)
    print(eval_tr)

    # %%
    test_lt_dict = get_longtail_dict(test)
    test_lt_labels = get_longtail_inputs(test, test_lt_dict)
    test_item_ids = np.array(test["movie_id"])
    test_sim_item_groups = test_user_hist

    pred_ts = model.predict(test_model_input,
                            test_lt_labels,
                            test_item_ids,
                            test_sim_item_groups,
                            batch_size=2000)
    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))

    # %%
    # 注释：7.Embedding
    # print("user embedding shape: ", model.user_dnn_embedding[:2])
    # print("item embedding shape: ", model.item_dnn_embedding[:2])

    # %%
    # 注释：8. get single tower
    # dict_trained = model.state_dict()  # trained model
    # trained_lst = list(dict_trained.keys())
    #    # # 注释：user tower
    # model_user = CDAN(user_feature_columns, [], popular_feature_columns, item_num, task='binary', device=device)
    # dict_user = model_user.state_dict()
    # for key in dict_user:
    #     dict_user[key] = dict_trained[key]
    # model_user.load_state_dict(dict_user)  # load trained model parameters of user tower
    # user_feature_name = user_sparse_features + user_dense_features + popular_dense_features
    # user_model_input = {name: test[name] for name in user_feature_name}
    # user_model_input["user_hist"] = test_user_hist
    # user_embedding = model_user.predict(user_model_input,
    #                                     test_lt_labels,
    #                                     test_item_ids,
    #                                     test_sim_item_groups,
    #                                     batch_size=2000)
    # # print("single user embedding shape: ", user_embedding[:2])
    #
    # # 注释：item tower
    # model_item = CDAN([], item_feature_columns, popular_feature_columns, item_num, task='binary', device=device)
    # dict_item = model_item.state_dict()
    # for key in dict_item:
    #     dict_item[key] = dict_trained[key]
    # model_item.load_state_dict(dict_item)  # load trained model parameters of item tower
    # item_feature_name = item_sparse_features + item_dense_features + popular_dense_features
    # item_model_input = {name: test[name] for name in item_feature_name}
    # item_model_input["genres"] = test_genres_list
    # item_embedding = model_item.predict(item_model_input,
    #                                     test_lt_labels,
    #                                     test_item_ids,
    #                                     test_sim_item_groups,
    #                                     batch_size=2000)
    # # print("single item embedding shape: ", item_embedding[:2])
