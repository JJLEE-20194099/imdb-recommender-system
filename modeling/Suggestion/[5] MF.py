import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import os

class MaxFactorization:
    def __init__(self, Y, K, lam=0.1, X = None, W=None, learning_rate=0.5, max_iter=1000, print_every=100, user_based = 1):
        self.Y_raw = Y
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_based = user_based
        self.n_users =1390
        self.n_items = 8352
        self.n_ratings = Y.shape[0]

        if X is None:
            self.X = np.random.randn(self.n_items, K)
        else:
            self.X = X
        
        if W is None:
            self.W =np.random.randn(K, self.n_users)
        else:
            self.W = W
        
        self.Y_normalize = self.Y_raw.copy()
    
    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users
        else:
            user_col = 1
            item_col = 0
            n_objects = self.n_items
        
        users = self.Y_raw[:, user_col]
        self.mean_users = np.zeros((n_objects,))

        for user in range(n_objects):
            ids = np.where(users == user)[0].astype(np.int32)
            item_ids = self.Y_normalize[ids, item_col]
            ratings = self.Y_normalize[ids, 2]

            m = np.mean(ratings)

            if np.isnan(m):
                m = 0
            self.mean_users[user] = m

            self.Y_normalize[ids, 2] = ratings - self.mean_users[user]
        

    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            user, item, rating = [int(self.Y_normalize[i, 0]), int(self.Y_normalize[i, 1]), int(self.Y_normalize[i, 2])]
            L += 0.5 * (rating - self.X[item, :].dot(self.W[:, user])) ** 2
        
        L /= self.n_ratings

        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro')) + np.linalg.norm(self.W, 'fro')
        return L



    def get_items_rated_by_user(self, user_id):
        ids = np.where(self.Y_normalize[:, 0] == user_id)[0]
        item_ids = self.Y_normalize[ids, 1].astype(np.int32)
        ratings = self.Y_normalize[ids, 2]
        return (item_ids, ratings)
    
    def get_users_who_rate_item(self, item_id):
        ids = np.where(self.Y_normalize[:, 1] == item_id)[0]
        user_ids = self.Y_normalize[ids, 0].astype(np.int32)
        ratings = self.Y_normalize[ids, 2]
        return (user_ids, ratings)
    
    def updateX(self):
        for item in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(item)
            W_item = self.W[:, user_ids]

            grad_x = -(ratings - self.X[item, :].dot(W_item)).dot(W_item.T) / self.n_ratings + \
                self.lam * self.X[item, :]
            self.X[item, :] -= self.learning_rate * grad_x.reshape((self.K,))

    def updateW(self):
        for user in range(self.n_users): 
            item_ids, ratings = self.get_items_rated_by_user(user)
            X_user = self.X[item_ids, :]

            grad_w = -X_user.T.dot(ratings - X_user.dot(self.W[:, user])) / self.n_ratings + \
                self.lam * self.W[:, user]
            
            self.W[:, user] -= self.learning_rate*grad_w.reshape((self.K, ))
    
    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw)
                print('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)
            
    def pred(self, u, i):
        u = int(u)
        i = int(i)

        if self.user_based:
            bias = self.mean_users[u]
        else:
            bias = self.mean_users[i]


        pred = self.X[i, :].dot(self.W[:, u]) +  bias   

        if pred < 0:
            return 0
        if pred > 5:
            return 5
        return pred
    
    def pred_for_user(self, user_id):
        ids = np.where(self.Y_normalize[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_normalize[ids, 1].tolist()
        y_pred = self.X.dot(self.W[:, user_id]) + self.mean_users[user_id]
        
        predicted_ratings = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings
    
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2]) ** 2
        
        RMSE = np.sqrt(SE/n_tests)
        return RMSE
    def pred_all(self, rate_test):
        res = []
        test = []
        n_tests = rate_test.shape[0]
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            res.append(pred)
            test.append(rate_test[n, 2])
        return [res, test]


class Metrics():
    def computeMSE(y_true, y_pred):
        return np.mean((y_true-y_pred)**2)

    def computeMAE(y_true, y_pred):
        return np.mean(np.abs(y_true-y_pred))

    def computeSIA(y_true, y_pred, eps=1): 
        error = np.abs(y_true - y_pred)
        bina = 0
        for err in error:
            if (err < eps):
                bina += 1
        return bina/len(y_true)

mae = 0
mse = 0
rmse = 0
sia_1 = 0
sia_0_5 = 0
sia_0_25 = 0

if not os.path.exists('../../result'):
    os.makedirs('../../result')
if not os.path.exists('../../result/MF'):
    os.makedirs('../../result/MF')

# for i in range(1, 6):
#     train_path = f'../../data/datasets/rating/kfold/u{i}.base.csv'
#     test_path = f'../../data/datasets/rating/kfold/u{i}.test.csv'
#     rate_train = pd.read_csv(train_path, sep=',', encoding='latin-1')[['user index', 'movie index', 'rating']].drop_duplicates(subset=['user index', 'movie index'], keep='first').values
#     rate_test = pd.read_csv(test_path, sep=',', encoding='latin-1')[['user index', 'movie index', 'rating']].drop_duplicates(subset=['user index', 'movie index'], keep='first').values
 
#     rs = MaxFactorization(rate_train, K = 10, lam=0.1, print_every=10, learning_rate=0.75, max_iter=100, user_based=1)
#     rs.fit()
#     Y_predict, Y_true = rs.pred_all(rate_test)
#     Y_predict = np.array(Y_predict)
#     Y_true = np.array(Y_true)

#     #Compute RMSE metrics
#     RMSE = rs.evaluate_RMSE(rate_test)
#     rmse += RMSE

#     #Compute MAE metrics
#     mae_test = Metrics.computeMAE(Y_true, Y_predict)
#     mae += mae_test

#     #Compute MSE metrics
#     mse_test = Metrics.computeMSE(Y_true, Y_predict)
#     mse += mse_test

#     #Compute SIA metrics
#     sia_test_1 = Metrics.computeSIA(Y_true, Y_predict, 1)
#     sia_1 += sia_test_1

#     sia_test_0_5 = Metrics.computeSIA(Y_true, Y_predict, 0.5)
#     sia_0_5 += sia_test_0_5

#     sia_test_0_25 = Metrics.computeSIA(Y_true, Y_predict, 0.25)
#     sia_0_25 += sia_test_0_25


#     with open(f'../../result/MF_fold_{i}.txt', "w+") as f:
#         for i in range(len(Y_predict)):
#             txt = str(Y_predict[i] ) + " " + str(Y_true[i]) + '\n'
#             f.write(txt)
#         f.write(f'RMSE: {RMSE}\n')
#         f.write(f'MAE: {mae_test}\n')
#         f.write(f'mse_test: {mse_test}\n')
#         f.write(f'SIA 1: {sia_test_1}\n')
#         f.write(f'SIA 0.5: {sia_test_0_5}\n')
#         f.write(f'SIA 0.25: {sia_test_0_25}\n')




        