from os import sep
import pandas as pd;
import numpy as np;
from scipy import sparse;
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import math
import seaborn as sns

class collaborative_filering:
    def __init__(self, Y, k_neighbors, distance_func=cosine_similarity, mode=1):
        self.mode = mode;
        if self.mode == 1:
            self.Y = Y
        else:
            self.Y = Y[:, [1, 0, 2]]

        self.k_neighbors = k_neighbors
        self.distance_func = distance_func
        self.Y_utility = None
        self.no_users = 1390
        self.no_movies = 8352

    def insert(self, data):
        self.Y = np.concatenate((self.Y, data), axis=0)

    def normalize_data(self):
        users = self.Y[:, 0]
        self.Y_utility = self.Y.copy()
        self.mean_user = np.zeros((self.no_users, ))
        for i in range(self.no_users):
            user_id_list = np.where(users == i)[0].astype(np.int32)
            item_id_list = self.Y[user_id_list, 1]
            rating_list = self.Y[user_id_list, 2]
            if (len(rating_list) == 0):
                m = 0
            else:
                m = np.mean(rating_list)
                if np.isnan(m):
                    m = 0
            self.mean_user[i] = m
            self.Y_utility[user_id_list, 2] = rating_list - self.mean_user[i]

        self.Y_utility_sparse = sparse.coo_matrix((self.Y_utility[:, 2], (
            self.Y_utility[:, 1], self.Y_utility[:, 0])),  (self.no_movies, self.no_users))
        self.Y_utility_sparse = self.Y_utility_sparse.tocsr()

    def cal_similarity(self):
        self.similarity_matrix = self.distance_func(
            self.Y_utility_sparse.T, self.Y_utility_sparse.T)

    def refresh(self):
        self.normalize_data()
        self.cal_similarity()

    def fit(self):
        self.refresh()

    def predict_utils(self, user, i):
        user = int(user)
        i = int(i)
        user_id_list = np.where(self.Y[:, 1] == i)[0].astype(np.int32)
        user_id_list = (self.Y[user_id_list, 0]).astype(np.int32)
        similar = self.similarity_matrix[user, user_id_list]

        k_user_id_nearest = np.argsort(similar)[-self.k_neighbors:]

        k_simlilar_nearest = similar[k_user_id_nearest]

        r = self.Y_utility_sparse[i, user_id_list[k_user_id_nearest]]

        return (r * k_simlilar_nearest)[0] / (np.abs(k_simlilar_nearest).sum() + 1e-8) + self.mean_user[user]

    def predict(self, user, i):

        if self.mode:
            return self.predict_utils(user, i)
        return self.predict_utils(i, user)

    def suggest(self, user):

        row_id_list = np.where(self.Y[:, 0] == user)[0]
        movie_list = self.Y[row_id_list, 1].tolist()

        res = []

        for i in range(self.no_movies):
            if i not in movie_list:
                rating = self.predict_utils(user, i)

                # if rating > self.mean_user[user]:
                #     res.append(i)
                if rating >= 4.5:
                    res.append(i)

        return res

    def print(self):
        suggest_out_path = '../../result/CF_suggest.csv'
        suggest_list = []
        if self.mode == 0:
            df = pd.DataFrame(columns=['movie index', 'user indexes'])
            df['movie index'] = np.zeros(self.no_users)
            df['user indexes'] = np.zeros(self.no_users)
            for i in range(self.no_users):
                res = self.suggest(i)
                res = [str(i) for i in res]
                suggest_list.append("|".join(res))
                # if self.mode:
                #     print ('Recommend item(s):', res, 'to user', i)
                # else:
                #     print ('Recommend item', i, 'to user(s) : ', res)
            df['user indexes'] = suggest_list
            df['movie index'] = range(self.no_users)
            df.to_csv(suggest_out_path, sep=',', index=False)


# neighbors = [1, 10, 20, 30, 50, 100]
# for neighbor in neighbors:
#     res = 0
#     cnt = 1
#     for i in range(1, 6):
#         train_path = f'../../data/datasets/rating/kfold/u{i}.base.csv'
#         test_path = f'../../data/datasets/rating/kfold/u{i}.test.csv'
#         rating_train = pd.read_csv(train_path, sep=',', encoding='latin-1')[['user index', 'movie index', 'rating']].drop_duplicates(subset=['user index', 'movie index'], keep='first').values
#         rating_test = pd.read_csv(test_path, sep=',', encoding='latin-1')[['user index', 'movie index', 'rating']].drop_duplicates(subset=['user index', 'movie index'], keep='first').values


#         CF_model = collaborative_filering(rating_train, k_neighbors=neighbor, mode=1)
#         CF_model.fit()

#         no_tests = rating_test.shape[0]
#         square_error = 0

#         Y_predict = []
#         for i in range(no_tests):
#             predict = CF_model.predict(rating_test[i, 0], rating_test[i, 1])
#             if (predict < 0):
#                 predict = 1
#             if (predict > 5):
#                 predict = 5
#             Y_predict.append(predict)
#             square_error += (predict - rating_test[i, 2]) ** 2

#         RMSE = np.sqrt(square_error/(no_tests))

#         print(RMSE)
#         res = res + RMSE
#         X = [i for i in range(no_tests)]

#         Y_true = rating_test[:, 2]
#         df = pd.DataFrame(columns=['True', 'Predict'])
#         df['True'] = Y_true
#         df['Predict'] = Y_predict
#         if not os.path.exists('../../result'):
#             os.makedirs('../../result')
#         if not os.path.exists('../../result/CF'):
#             os.makedirs('../../result/CF')
#         if not os.path.exists('../../result/CF/user_vote_item_mode_1'):
#             os.makedirs('../../result/CF/user_vote_item_mode_1')
        
#         df.to_csv(f'../../result/CF/user_vote_item_mode_1/CF_fold_{neighbor}_{cnt}.txt', sep=',', index=False)
#         cnt += 1

#     print('neighbor: ', neighbor, '-', 'RMSE: ', res / 5)

class Metrics():
    def computeMSE(y_true, y_pred):
        return np.mean((y_true-y_pred)**2)

    def computeMAE(y_true, y_pred):
        return np.mean(np.abs(y_true-y_pred))

    def computeSIA(y_true, y_pred, eps=1): 
        error = np.abs(y_true - y_pred)
        bina = [1 if err <= eps else 0 for err in error]
        res = np.mean(bina)
        return res

avg_mae_test = 0
avg_mse_test = 0
avg_sia_1_test = 0
avg_sia_0_5_test = 0
avg_sia_0_25_test = 0

avg_mae_train = 0
avg_mse_train = 0
avg_sia_1_train = 0
avg_sia_0_5_train = 0
avg_sia_0_25_train = 0
for i in range(1, 6):
    train_path = f'../../data/datasets/rating/kfold/u{i}.base.csv'
    test_path = f'../../data/datasets/rating/kfold/u{i}.test.csv'
    rating_train = pd.read_csv(train_path, sep=',', encoding='latin-1')[['user index', 'movie index', 'rating']].drop_duplicates(subset=['user index', 'movie index'], keep='first').values
    rating_test = pd.read_csv(test_path, sep=',', encoding='latin-1')[['user index', 'movie index', 'rating']].drop_duplicates(subset=['user index', 'movie index'], keep='first').values
    res = 0    

    CF_model = collaborative_filering(rating_train, k_neighbors=50, mode=1)
    CF_model.fit()
    no_tests = rating_test.shape[0]
    Y_predict_test = []
    for i in range(no_tests):
        predict = CF_model.predict(rating_test[i, 0], rating_test[i, 1])
        if (predict < 1):
            predict = 1
        if (predict > 5):
            predict = 5

        Y_predict_test.append(predict)
    X = [i for i in range(no_tests)]
    Y_true = rating_test[:, 2]


    #Compute MAE metrics
    mae_test = Metrics.computeMAE(Y_true, Y_predict_test)
    avg_mae_test += mae_test

    #Compute MSE metrics
    mse_test = Metrics.computeMSE(Y_true, Y_predict_test)
    avg_mse_test += mse_test

    #Compute SIA metrics
    sia_test_1 = Metrics.computeSIA(Y_true, Y_predict_test, 1)
    avg_sia_1_test += sia_test_1

    sia_test_0_5 = Metrics.computeSIA(Y_true, Y_predict_test, 0.5)
    avg_sia_0_5_test += sia_test_0_5

    sia_test_0_25 = Metrics.computeSIA(Y_true, Y_predict_test, 0.25)
    avg_sia_0_25_test += sia_test_0_25

    no_trains = rating_train.shape[0]
    Y_predict_train = []
    for i in range(no_trains):
        predict = CF_model.predict(rating_train[i, 0], rating_train[i, 1])
        if (predict < 1):
            predict = 1
        if (predict > 5):
            predict = 5

        Y_predict_train.append(predict)
    X = [i for i in range(no_trains)]
    Y_true = rating_train[:, 2]


    #Compute MAE metrics
    mae_train = Metrics.computeMAE(Y_true, Y_predict_train)
    avg_mae_train += mae_train

    #Compute MSE metrics
    mse_train = Metrics.computeMSE(Y_true, Y_predict_train)
    avg_mse_train += mse_train

    #Compute SIA metrics
    sia_train_1 = Metrics.computeSIA(Y_true, Y_predict_train, 1)
    avg_sia_1_train += sia_train_1

    sia_train_0_5 = Metrics.computeSIA(Y_true, Y_predict_train, 0.5)
    avg_sia_0_5_train += sia_train_0_5

    sia_train_0_25 = Metrics.computeSIA(Y_true, Y_predict_train, 0.25)
    avg_sia_0_25_train += sia_train_0_25

print("mae_train:", avg_mae_train / 5)
print("mse_train:", avg_mse_train / 5)
print("sia_1_train:", avg_sia_1_train / 5)
print("sia_0_5_train:", avg_sia_0_5_train / 5)
print("sia_0_25_train:", avg_sia_0_25_train / 5)

print("mae_test:", avg_mae_test / 5)
print("mse_test:", avg_mse_test / 5)
print("sia_1_test:", avg_sia_1_test / 5)
print("sia_0_5_test:", avg_sia_0_5_test / 5)
print("sia_0_25_test:", avg_sia_0_25_test / 5)

# CF_model.print()






