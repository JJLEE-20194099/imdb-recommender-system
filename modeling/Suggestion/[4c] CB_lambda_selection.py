import pandas as pd;
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np;
from sklearn.linear_model import Ridge
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
import matplotlib.pyplot as plt5
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
import math
from sklearn.preprocessing import StandardScaler

movies = pd.read_csv('../../data/datasets/movie/ml_detail.csv', sep=',', encoding='latin-1')
movies = movies.drop(columns=['movie index'])

no_movies = movies.shape[0]
print('No movie themes: ', no_movies)
print(movies.shape)
X_train = movies[["Reality-TV","News","War", "Musical","Sci-Fi","Film-Noir","Thriller","Action","Biography","Family","Game-Show","Music","Short","Adventure","Animation","History","Drama","Horror","Documentary","Mystery","Western","Fantasy","Comedy","Sport","Talk-Show","Crime","Romance"]].values[:,:]
transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train).toarray()

no_movie_theme = tfidf.shape[1]
n_users = 1390

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

def get_movies_rated_by_user(utility_matrix, user_index):
    user_index_list = utility_matrix[:, 6]
    row_ids = np.where(user_index_list == user_index)[0]
    movie_index_util_list = utility_matrix[row_ids, 5]
    rating_list = utility_matrix[row_ids, 2]
    return (movie_index_util_list, rating_list)

avg_mae_tests = []
avg_mse_tests = []
avg_rmse_tests = []
avg_sia_1_tests = []
avg_sia_0_5_tests = []
avg_sia_0_25_tests = []

avg_mae_trains = []
avg_mse_trains = []
avg_rmse_trains = []
avg_sia_1_trains = []
avg_sia_0_5_trains = []
avg_sia_0_25_trains = []

for lamda in tqdm([.1, 1, 5, 10, 100, 500, 1000]):
    avg_mae_test = 0
    avg_mse_test = 0
    avg_rmse_test = 0
    avg_sia_1_test = 0
    avg_sia_0_5_test = 0
    avg_sia_0_25_test = 0

    avg_mae_train = 0
    avg_mse_train = 0
    avg_rmse_train = 0
    avg_sia_1_train = 0
    avg_sia_0_5_train = 0
    avg_sia_0_25_train = 0

    for i in range(1, 6):
        train_path = f'../../data/datasets/rating/kfold/u{i}.base.csv'
        test_path = f'../../data/datasets/rating/kfold/u{i}.test.csv'
        ratings_base = pd.read_csv(train_path, sep=',', encoding='latin-1').drop_duplicates(subset=['user index', 'movie index'], keep='first')
        ratings_test = pd.read_csv(test_path, sep=',', encoding='latin-1').drop_duplicates(subset=['user index', 'movie index'], keep='first')
        ratings_base.columns= ['movie id','user id', 'rating', 'comment', 'date', 'movie index', 'user index']
        ratings_base = pd.merge(ratings_base, movies, how='inner', on='movie id')
        ratings_test = pd.merge(ratings_test, movies, how='inner', on='movie id')
        # print(type(ratings_base))
        ratings_train_arr = ratings_base.values[1:, :]
        # print(type(ratings_train_arr))
        ratings_test_arr = ratings_test.values[1:, :]
        user_index_list = ratings_base['user index'].value_counts().index.tolist()

        w = np.zeros((no_movie_theme, n_users))
        b = np.zeros((1, n_users))

        for i in user_index_list:
            movie_index_util_list, rating_list = get_movies_rated_by_user(ratings_train_arr, i)
            ridge = Ridge(alpha=lamda, fit_intercept=True, max_iter=20000, normalize=True)
            model = make_pipeline(StandardScaler(with_mean=False), Ridge())
            tfdif_by_user = tfidf[movie_index_util_list.tolist()]
            
            ridge.fit(tfdif_by_user, rating_list)

            w[:, i] = ridge.coef_
            b[0, i] = ridge.intercept_
        
        Y = tfidf.dot(w)  + b

        movie_indexes = ratings_train_arr[:, 5]
        user_indexes = ratings_train_arr[:, 6]
        true_scores = ratings_train_arr[:, 2]

        y_true_train = []
        y_predict_train = []
        for i in range(len(movie_indexes)):
            movie_index = movie_indexes[i]
            user_index = user_indexes[i]
            true_score = true_scores[i]
            if (Y[movie_index][user_index] != 0):
                y_true_train.append(true_score)
                y_predict_train.append(Y[movie_index][user_index])
        y_predict_train = [5 if y > 5 else y for y in y_predict_train]
        y_predict_train = [1 if y < 0 else y for y in y_predict_train]
        y_predict_train = np.array(y_predict_train)
        y_true_train = np.array(y_true_train)

        #Compute MAE metrics
        mae_train = Metrics.computeMAE(y_true_train, y_predict_train)
        avg_mae_train += mae_train
        #Compute MSE metrics
        mse_train = Metrics.computeMSE(y_true_train, y_predict_train)
        avg_mse_train += mse_train
        #Compute SIA metrics
        sia_1_train = Metrics.computeSIA(y_true_train, y_predict_train, 1)
        avg_sia_1_train += sia_1_train

        sia_0_5_train= Metrics.computeSIA(y_true_train, y_predict_train, 0.5)
        avg_sia_0_5_train += sia_0_5_train

        sia_0_25_train = Metrics.computeSIA(y_true_train, y_predict_train, 0.25)
        avg_sia_0_25_train += sia_0_25_train

        movie_indexes = ratings_test_arr[:, 5]
        user_indexes = ratings_test_arr[:, 6]
        true_scores = ratings_test_arr[:, 2]

        y_true_test = []
        y_predict_test = []
        import math
        for i in range(len(movie_indexes)):
            movie_index = movie_indexes[i]
            user_index = user_indexes[i]
            true_score = true_scores[i]
            if (Y[movie_index][user_index] != 0):
                y_true_test.append(true_score)
                y_predict_test.append(Y[movie_index][user_index])
        y_predict_test = [5 if y > 5 else y for y in y_predict_test]
        y_predict_test = [1 if y < 0 else y for y in y_predict_test]
        y_predict_test = np.array(y_predict_test)
        y_true_test = np.array(y_true_test)

        #Compute MAE metrics
        mae_test = Metrics.computeMAE(y_true_test, y_predict_test)
        avg_mae_test += mae_test
        #Compute MSE metrics
        mse_test = Metrics.computeMSE(y_true_test, y_predict_test)
        avg_mse_test += mse_test
        #Compute SIA metrics
        sia_1_test = Metrics.computeSIA(y_true_test, y_predict_test, 1)
        avg_sia_1_test += sia_1_test

        sia_0_5_test= Metrics.computeSIA(y_true_test, y_predict_test, 0.5)
        avg_sia_0_5_test += sia_0_5_test

        sia_0_25_test = Metrics.computeSIA(y_true_test, y_predict_test, 0.25)
        avg_sia_0_25_test += sia_0_25_test



    avg_mae_trains.append(avg_mae_train / 5)
    avg_mse_trains.append(avg_mse_train / 5)
    avg_sia_1_trains.append(avg_sia_1_train / 5)
    avg_sia_0_5_trains.append(avg_sia_0_5_train / 5)
    avg_sia_0_25_trains.append(avg_sia_0_25_train / 5)

    avg_mae_tests.append(avg_mae_test / 5)
    avg_mse_tests.append(avg_mse_test / 5)
    avg_sia_1_tests.append(avg_sia_1_test / 5)
    avg_sia_0_5_tests.append(avg_sia_0_5_test / 5)
    avg_sia_0_25_tests.append(avg_sia_0_25_test / 5)

X = [.1, 1, 5, 10, 100, 500, 1000]
plt1.scatter(X, avg_mae_trains)
plt1.plot(X, avg_mae_tests, color="orange", marker="o")
plt1.xlabel("learning rate")
plt1.ylabel("mae metric")
plt1.savefig('../../result/images/CB/ridge_lamda/mae.png')
# plt2.scatter(X, avg_mse_trains)
# plt2.plot(X, avg_mse_tests, color="orange", marker="o")
# plt2.xlabel("learning rate")
# plt2.ylabel("mse metric")
# plt2.savefig('../../result/images/CB/ridge_lamda/mse.png')
# plt3.scatter(X, avg_sia_1_trains)
# plt3.plot(X, avg_sia_1_tests, color="orange", marker="o")
# plt3.xlabel("learning rate")
# plt3.ylabel("sia ep=1 metric")
# plt3.savefig('../../result/images/CB/ridge_lamda/sia_1.png')
# plt4.scatter(X, avg_sia_0_5_trains)
# plt4.plot(X, avg_sia_0_5_tests, color="orange", marker="o")
# plt4.xlabel("learning rate")
# plt4.ylabel("sia ep=0.5 metric")
# plt4.savefig('../../result/images/CB/ridge_lamda/sia_0_5.png')
# plt5.scatter(X, avg_sia_0_25_trains)
# plt5.plot(X, avg_sia_0_25_tests, color="orange", marker="o")
# plt5.xlabel("learning rate")
# plt5.ylabel("sia ep=0.25 metric")
# plt5.savefig('../../result/images/CB/ridge_lamda/sia_0_25.png')