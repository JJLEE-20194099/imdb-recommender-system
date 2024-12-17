# Film Recommendation System - Machine Learning and Data Mining

***Note:*** Due to the inclusion of some formulas, we recommend using the following file for a complete and accurate view:  
[https://hackmd.io/@jjleeblogger/BJLNHhC4Je](https://hackmd.io/@jjleeblogger/BJLNHhC4Je)

- [Machine Learning and Data Mining](#)
  * [Contributors](#)
  * [Work Allocation](#)
- [Introduction to the Topic](#)
- [Data Collection](#)
- [Data Integration](#)
- [Data Cleaning and Preprocessing](#)
- [IMDB Datasets](#)
- [Data Exploration](#)
- [Key Concepts and Metrics for the Problem](#)
- [Models](#)

  * [Collaborative Filtering](#)
      + [Similarity Function](#)
          + [Data Normalization](#)
          + [Cosine Similarity Function](#)
      + [Model](#)
          + [User-Item](#)
          + [Item-User](#)
      + [Model Evaluation](#)
          + [MSE](#)
          + [MAE](#)
          + [SIA 1](#)
          + [SIA 0.5](#)
          + [SIA 0.25](#)

  * [Matrix Factorization](#)
      + [Matrix Decomposition](#)
          + [Latent User and Movie Matrices](#)
          + [Meaning of Matrices](#)
      + [Loss Function](#)
      + [Model Evaluation](#)
          + [MSE](#)
          + [MAE](#)
          + [SIA 1](#)
          + [SIA 0.5](#)
          + [SIA 0.25](#)

  * [Content-Based](#)
      + [Building Feature Vectors with TF-IDF](#)
      + [Constructing a Model for Each User](#)
      + [Model Evaluation](#)
          + [MSE](#)
          + [MAE](#)
          + [SIA 1](#)
          + [SIA 0.5](#)
          + [SIA 0.25](#)

  * [Applying Simple Neural Networks for Rating Prediction](#)
      + [Using MLP and GMF in Matrix Factorization](#)
          + [Concept](#)
          + [GMF](#)
              + [Model](#)
              + [Model Evaluation](#)
          + [MLP](#)
              + [Model](#)
              + [Model Evaluation](#)
          + [GMF and MLP](#)
              + [Model](#)
              + [Model Evaluation](#)

  * [Sentiment Analysis Based on Review Texts](#)
      + [Review Content Preprocessing](#)
      + [Input Construction](#)
      + [Building a Simple Neural Network Model](#)
      + [Model Evaluation](#)
          + [Confusion Matrix](#)
          + [Precision, Recall](#)

- [Program Instructions](#)
  * [Data Collection](#)
  * [Data Cleaning and Preprocessing](#)
  * [Data Exploration](#)
  * [Model Testing](#)
- [Demo](#)

## Self-Assessment of the Machine Learning and Data Mining Project

| Task | Status |
|------|--------|
| Data collection using the BeautifulSoup library | Completed |
| Data integration | Completed |
| Data cleaning and preprocessing | Completed |
| Data exploration and analysis of movies and ratings | Completed |
| Sentiment analysis on review texts for satisfaction levels (1-2-3 stars vs. 4-5 stars) | Completed |
| Development of three main models: collaborative filtering, content-based, and matrix factorization | Completed |

**Outcome:**  
The group gained a comprehensive understanding of various rating prediction methods, providing insights to address the movie recommendation problem for users.

---

## Contributors

| Name | Student ID |
|------|------------|
| Lê Thành Long | 20194099 |
| Phạm Thế Nam | 20190058 |
| Đỗ Mạnh Quân | 20194143 |
| Lê Đình Nam | 20194124 |

***This project demonstrates how our group built a film recommender system for the IMDb website. It is a machine learning and data mining project conducted at our university.***  

***Our system is named ImdbFilm_Recommender.***  

***@Author: LNQ GROUP***

---

## Work Distribution within the Group

| Name | Task |
|------|------|
| Lê Thành Long | Implemented data crawling from [IMDb](https://www.imdb.com/), collaborative filtering, and several neural network models to enhance results. |
| Phạm Thế Nam | Researched and implemented matrix factorization, and prepared presentation slides. |
| Đỗ Mạnh Quân | Researched and implemented the content-based recommendation method, and developed a demo interface. |
| Lê Đình Nam | Integrated, cleaned, and preprocessed data, including text preprocessing for review analysis. |
| Entire Group | Data visualization, contributing ideas to refine code, report, and presentation slides. |

---

## Project Introduction

Attracting users to a system is a vital goal for any modern platform. To achieve this, the system must provide a comfortable and convenient user experience. Our group explored the problem of predicting ratings and recommending movies to users, aiming to suggest films that users are likely to enjoy. This approach optimizes users' search time and enhances their satisfaction with the system.

We focused on three main methods: collaborative filtering, matrix factorization, and content-based recommendations.  

Additionally, we applied GMF and MLP models to improve matrix factorization (details are presented at the end of the project).  

To evaluate the models, we utilized various metrics, identifying the most meaningful ones for rating prediction and movie recommendation.

## Data Collection

Our group collected data from the renowned movie review website [IMDb](https://www.imdb.com/). While many datasets related to IMDb are already available online, our dataset stands out for its richness in detailed movie information and ratings. Notably, the movies in our dataset are all recent, which means the associated reviews are also up-to-date.

***Library used for data collection: BeautifulSoup***  
(The data collection script can be found in the `src` directory of the project.)

---

## Data Integration

During the data crawling process, we identified approximately 14 main movie genres, each associated with specific IDs.  

We integrated the genre files and continued to collect detailed data about movies and their ratings.

---

## Data Cleaning and Preprocessing

Data collection inevitably involves errors and missing values. Additionally, IMDb implements anti-crawling mechanisms, which led to certain missing values in our dataset.

***Basic steps for data cleaning and preprocessing:***

- Removed movies with `NaN` titles.
- For users who rated the same movie more than once, only the first rating was kept.
- Eliminated users who rated either excessively or too sparingly.
- Standardized the IDs of users and movies.
- Since a movie can belong to multiple genres (27 genres in total), we created 27 additional columns, each representing a genre. A value of 1 indicates that the movie belongs to that genre, while a value of 0 indicates otherwise.
- Removed duplicate movies, users, and ratings.

---

## IMDb Datasets

Our project uses three main datasets:  
- `movie/ml_detail.csv`: Detailed movie information.
- `rating/ml_detail.csv`: Reviews and ratings for movies.
- `user/ids`: User IDs.  

### Overview and Description of the Datasets:

#### **`movie/ml_detail.csv`**  
**Dimensions:** 8352 rows, 45 columns  

|    | Column            | Dtype   | Value Range   | Description                                                                                  |
|---:|:------------------|:--------|:--------------|:---------------------------------------------------------------------------------------------|
|  0 | `movie id`        | string  | `NaN`         | The ID of the movie.                                                                         |
|  1 | `title`           | string  | `NaN`         | The title of the movie.                                                                      |
|  2 | `series`          | string  | `NaN`         | Indicates whether the movie is part of a series or a standalone film.                        |
|  3 | `release year`    | int64   | `>= 0`        | The year the movie was released.                                                            |
|  4 | `certification`   | string  | `NaN`         | The certification tag of the movie (e.g., parental guidance, adult content).                |
|  5 | `duration`        | string  | `NaN`         | The runtime of the movie.                                                                    |
|  7 | `average rating`  | float   | `>= 0`        | The average rating of the movie.                                                            |
|  8 | `rating total`    | string  | `NaN`         | The total number of ratings the movie received.                                             |
|  9 | `genre list`      | string  | `NaN`         | The list of genres the movie belongs to.                                                    |
| 10 | `content`         | string  | `NaN`         | A brief synopsis of the movie.                                                              |
| 11 | `...`             | `...`   | `...`         | Refer to the dataset for additional columns and details.                                     |

---

#### **`rating/ml_detail.csv`**  
**Dimensions:** 93,246 rows, 9 columns  

|    | Column            | Dtype   | Value Range   | Description                                                                                  |
|---:|:------------------|:--------|:--------------|:---------------------------------------------------------------------------------------------|
|  0 | `movie id`        | string  | `NaN`         | The ID of the rated movie.                                                                   |
|  1 | `name`            | string  | `NaN`         | The title of the movie.                                                                      |
|  2 | `user id`         | string  | `NaN`         | The ID of the user who left the review.                                                     |
|  3 | `rating`          | int64   | `1-5`         | The user's rating for the movie.                                                            |
|  4 | `content`         | string  | `NaN`         | The review content.                                                                         |
|  5 | `date`            | string  | `NaN`         | The date the review was submitted.                                                          |
|  7 | `user index`      | int64   | `0-1389`      | The user's index after preprocessing and standardizing the `user id`.                       |
|  8 | `movie index`     | int64   | `0-8351`      | The movie's index after preprocessing and standardizing the `movie id`.                     |

---

**Note:**  
We split the `rating/ml_detail.csv` dataset using ***k-fold cross-validation (k = 5)*** and averaged the results across all folds.

## Data Exploration:

The group highlights a few key findings from the data exploration phase. For more details, refer to the project file ***[4] Data Exploration.ipynb***.

- The percentage of viewers for comedy and romance movies is the highest, and their average ratings are also relatively high.
- The average length of reviews before preprocessing is approximately 1500 characters.
- Most ratings fall in the range of 3-4-5.
- ...

---

## Key Concepts and Metrics Used in the Project

### SIA - Soft Interval Accuracy  
A prediction is considered correct if the difference between the predicted and actual values is within a hyperparameter epsilon:  
$$|y_i - \hat{y_i}| \leq \epsilon$$  
***SIA*** is then defined as the ratio of correctly predicted samples to the total number of samples.

---

### MAE - Mean Absolute Error  
$$MAE(y, \hat{y}) = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y_i}|$$

---

### MSE - Mean Squared Error  
$$MSE(y, \hat{y}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y_i})^2$$

---

### RMSE - Root Mean Squared Error  
$$RMSE(y, \hat{y}) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y_i})^2}$$

---

### Confusion Matrix  
Used primarily for classification problems, this is a square matrix with dimensions equal to ***number of classes × number of classes***.  
The value at row \(i\), column \(j\) represents the proportion of samples ***belonging to class \(i\) but predicted as class \(j)\***.

---

### Precision and Recall  

For binary classification problems:
- **Precision:** The ratio of correctly predicted samples for class 1 to the total samples predicted as class 1.  
- **Recall:** The ratio of correctly predicted samples for class 1 to the total samples that truly belong to class 1.  

***Key Notes:***
- High precision implies a low likelihood of false positives.
- High recall implies a low likelihood of missing true positives.

---

## Model

### Collaborative Filtering  
This project utilizes **Collaborative Filtering** for prediction, evaluated using metrics like ***Mean Absolute Error (MAE)***, ***Soft Interval Accuracy (SIA)***, or ***Root Mean Square Error (RMSE)***.

#### Features Used for Prediction  

| Feature Name | Type       |  
|--------------|------------|  
| `movie index` | numerical |  
| `user index`  | numerical |  
| `rating`      | numerical |  

---

### Similarity Function  

#### Data Normalization  
- **User-item matrix:** Each cell represents the rating of a user for a specific movie. If a movie has not been rated by a user, the value is set to the average of the movies rated by that user.  
- **Normalization:** For each rating, subtract the mean rating of the corresponding user. After normalization, the matrix contains positive, negative, and zero values, where zero indicates an unrated movie.

---

#### Cosine Similarity  

Each column in the matrix represents a user vector. Given two user vectors \(u_1\) and \(u_2\), their similarity is calculated as:  
$$\text{cosine_similarity}(u_1, u_2) = \frac{u_1^T u_2}{\|u_1\|_2 \cdot \|u_2\|_2}$$  

- **Value range:** [-1, 1]  
  - \(1\): Behavior of the two users is identical.  
  - \(-1\): Behavior of the two users is entirely opposite.

---

### Model Description  

***Note:*** The group explains the user-user model here; the item-item model follows a similar logic.

1. Compute the similarity between users to form a square matrix with dimensions equal to the number of users. The diagonal values of the matrix are all \(1\).
2. Using the idea of K-Nearest Neighbors (KNN), select \(k\) users most similar to the target user.
3. Predict the rating of a target user for a movie they have not rated based on these \(k\) nearest neighbors.

---

#### Prediction Formula  

![](https://i.imgur.com/jDzadkS.png)  

Where:
- \(N(u, i)\) is the set of \(k\) users most similar to user \(u\) who have rated movie \(i\).  
- \(\text{sim}(u, u')\) is the similarity between user \(u\) and user \(u'\).  
- \(\overline{r}_{u'}\) is the mean rating of user \(u'\).  

---

### Example Calculation  

To predict the rating of **user \(u_0\)** for **movie \(i_0\):**
1. Consider other users who have rated \(i_0\): \({u_1}, {u_2}, {u_4}, {u_5}\).
2. Compute the similarities of \(u_0\) with these users using cosine similarity:  
   **0.23, 0.45, -0.56, -0.1**.
3. Set \(k = 3\) and select the top 3 most similar users: \({u_1}, {u_2}, {u_5}\).
4. Retrieve the normalized ratings of these users for \(i_0\): **0.25, 0.75, 1.25**.
5. Compute the predicted rating:  
   $$\hat{y}_{i_0, u_0} = \frac{(0.23 \cdot 0.25) + (0.45 \cdot 0.75) + (-0.1 \cdot 1.25)}{|0.23| + |0.45| + |-0.1|} = 0.3461$$  

---

### Hyperparameter \(k\):  
The group tested various values of \(k\) and found \(k = 50\) to work best across most metrics with the crawled dataset.

#### Model Evaluation

Results:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.44|0.28|0.98|0.59|0.31|
|test|0.77|0.99|0.72|0.42|0.23|

***Comments***: The model achieves good accuracy. The performance on the `val` and `test` sets differs slightly for the MAE and MSE metrics. For the SIA metric, we observe that 72% of the test samples are predicted within an acceptable range with an error of 1, and the SIA 0.5 and SIA 0.25 scores are quite low on the test set. The team believes that movie ratings are hard to determine, especially when distinguishing between ratings like 2 or 3, so we decided to use the SIA 1 metric as the main evaluation metric for future models, with MAE and MSE as secondary metrics for monitoring.

---

### Matrix Factorization Method

#### Matrix Factorization into Factors

##### Approximation with Two Latent Matrices

***Idea***: Given a user-item matrix where each entry represents a rating given by a user to an item, some entries may be missing because users have not rated some movies. We approximate this matrix using two latent matrices, called the user latent matrix and the movie latent matrix.

Let matrix R represent the user-item matrix, which we approximate by the product of two latent matrices, P and Q.

Matrix R has dimensions m x n, where m is the number of users and n is the number of movies. Let the two latent matrices be P (m x k) and Q (n x k), where k << m, n.

Then:

\begin{equation}
\hat{R}  = PQ^T
\end{equation}

Specifically, for the rating of user u on movie i, we have:
\begin{equation}
\hat{R_\text{ui}}  = p_u q_i^T
\end{equation}

##### Meaning of the Two Latent Matrices

The user latent matrix P contains rows where each row p_u represents the preferences of user u for various movie features such as genre, content, etc.

The movie latent matrix Q contains rows where each row q_i represents the level of movie i's ownership of certain features, such as title, genre, or content.

***Note***: The formulas above include a bias term bu. Why do we need to add a bias for each user?

This is because people's movie-watching habits can vary, affecting how they rate a movie.

***Reason:*** Some people are more lenient and might rate a good movie 5 stars, while others may rate it 4 or even 3 stars.

To address this, our team calculates the bias for each user as the average rating given by that user to the movies they have watched. If we set the bias to 2.5, for example, this might not be accurate for more critical users.

#### Loss Function

The loss function used by our team is the average squared difference between predicted and actual ratings, with L2 regularization to prevent overfitting.

$\text{RSS}(y, \hat{y}) = {\frac{1}{m} \sum\limits_{i=1}^{m} (y_i - \hat{y}_i)^2}$

$Loss(y, \hat{y}) = \text{RSS}(y, \hat{y}) + \text{Regularization_Func(P, Q)}$

***Note***: Our team chose the regularization function:

$\text{Regularization_Func(P, Q)} = \frac{1}{2} * {\lambda} *  (\sqrt{\sum\limits_{i=1}^m \sum\limits_{j=1}^k |p_{ij}|^2} + \sqrt{\sum\limits_{i=1}^n \sum\limits_{j=1}^k |q_{ij}|^2} )$

#### Model Evaluation

Results:
* ***When lambda=0***

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|1.46|3.48|0.48|0.23|0.1430|
|test|1.47|3.55|0.48|0.23|0.1438|

* ***When lambda=0.1***

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.71|0.42|0.23|
|test|0.78|1.01|0.70|0.41|0.22|

* ***When lambda=0.5***

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

* ***When lambda=1***

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

***Comments***:

When ***lambda=0***, the model's accuracy is quite low for both the train and test sets, with the train set being more accurate than the test set. Removing the L2 regularization results in poor learning.

When ***lambda != 0***, the model achieves good accuracy, with minimal difference between the train and test sets. The performance on the `val` and `test` sets differs slightly for the MAE and MSE metrics.

For ***lambda=0.1, 0.5, 1***, the results are nearly identical, so we choose ***lambda=0.5*** as the final value.

For the SIA metric, we see that 70% of the test samples are predicted within an acceptable range with an error of 1, with the SIA 0.5 and SIA 0.25 scores being quite low on the test set. The model performs better when L2 regularization is added. The team believes that rating movies can be difficult due to the subjective nature of reviews, so we decided to use SIA 1 as the main metric for future models, alongside MAE and MSE for monitoring.

***With lambda=0.5:***
* learning_rate=0.1:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.71|0.42|0.23|
|test|0.78|1.01|0.70|0.41|0.22|

* learning_rate=1:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.71|0.42|0.23|
|test|0.78|1.01|0.70|0.41|0.22|

* learning_rate=0.5:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

* learning_rate=0.75:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

* ***With lambda = 0.5, learning_rate = 0.75:***

* When k = 5:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

* When k = 7:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

* When k = 10:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

* When k = 20:

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

### Content-Based Recommendation Method
#### Constructing Feature Vector Using TF-IDF

In the dataset movie/ml_detail.csv, we have 27 columns representing genres, and each movie can belong to multiple genres.

***Example:*** A movie belongs to the genres ***Animation|Comedy|Family***, so the columns for Animation, Comedy, and Family have a value of 1, while the other 24 genre columns have a value of 0.

We will construct a feature vector matrix using TF-IDF, where each row of the matrix represents the feature vector for each movie.

#### Building a Model for Each User

After obtaining the feature vector matrix, we will build a model for each user. In this case, our team uses a simple regression model. Specifically, we use models like Linear Regression and add a Ridge Regression model.

#### Error Function

$\text{RSS}(y, \hat{y}) = {\frac{1}{m} \sum\limits_{i=1}^{m} (y_i - \hat{y}_i)^2}$

$Loss(y, \hat{y}) = \text{RSS}(y, \hat{y}) + \text{Regularization_Func}$

***Inputs:***

* The input X for user u is the feature extractor of the movies that u has rated.
* The input y for user u is the rating scores for the movies that u has watched.

#### Model Evaluation

Results:

* Linear Regression Model

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.60|0.71|0.78|0.53|0.35|
|test|0.88|1.39|0.67|0.39|0.23|

* Ridge Regression Model:
    + ***(${\lambda}$=0.1)***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6292|0.7256|0.7788|0.5168|0.3239|
        |test|0.8327|1.2132|0.6824|0.4031|0.2268|

    + ***(${\lambda}$=1)***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6816|0.7916|0.7583|0.4727|0.2627|
        |test|0.7945|1.0689|0.6967|0.4119|0.2246|
    
    + ***(${\lambda}$=5)***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.7299|0.8849|0.7296|0.4372|0.2365|
        |test|0.7812|1.0183|0.7024|0.4129|0.2225|
    
    + ***(${\lambda}$=10)***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.7436|0.9165|0.7217|0.4298|0.2315|
        |test|0.7798|1.0129|0.7028|0.4132|0.2226|
        
    + ***(${\lambda}$=100)***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.7601|0.9581|0.7137|0.4223|0.2269|
        |test|0.7791|1.0106|0.7037|0.4125|0.2226|
    
     + ***(${\lambda}$=500)***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.7618|0.9627|0.7136|0.4221|0.2267|
        |test|0.7791|1.0106|0.7037|0.4124|0.2225|
    
    + ***(${\lambda}$=1000)***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.7618|0.9627|0.7136|0.4221|0.2267|
        |test|0.7791|1.0106|0.7037|0.4124|0.2225|
    
        
            

***Observations:***

Based on the main metric used by our team, which is SIA 1, we see that the Linear Regression model predicts 67% of the test samples within an acceptable range. Meanwhile, for Ridge with ${\lambda} = 5, 100, 500, 1000$, around 70% of the test samples are predicted within the acceptable range.

Ridge regression seems more effective than linear regression, mainly because the alpha hyperparameter helps prevent overfitting.

---

### Application of Simple Neural Networks for Rating Prediction

#### Applying MLP and GMF to Matrix Factorization Method

##### Idea

As you know, the matrix factorization method we mentioned earlier finds two latent matrices for users and movies such that their product approximates the rating matrix as closely as possible.

The simplicity of the dot product of two matrices makes it difficult to learn the latent features of both users and movies.

Therefore, in this section, our team introduces a neural matrix factorization network called NeuMF. We are using neural networks here due to their flexibility and non-linearity.

In this section, we will focus on two sub-networks: ***Generalized Matrix Factorization (GMF)*** and ***MLP***. 

We will use each network individually and combine them to evaluate their performance in the next section.

##### GMF - Generalized Matrix Factorization

###### Model
The GMF sub-network is similar to the matrix factorization method we discussed earlier with two latent matrices: the user embedding matrix and the movie embedding matrix, with sizes of ***(num_users + 1) * latent_dim and (num_items + 1) * latent_dim*** respectively.

***Where:*** num_users and num_items are the number of users and items, and latent_dim is the number of hidden features of the movies and users you want your model to learn.

***Why add 1 to num_users and num_items?***

As we know, the system does not have a fixed number of users and items. Therefore, when a new user or item appears that has not been encountered before, it will be assigned an ***<OOV>*** value (similar to a word that does not exist in a dictionary).

***Note:*** In the algorithm implementation, our team may add an ***Embedding regularizer l2*** to prevent overfitting for both users and items.

After creating two latent matrices for users and items, we expand these matrices and perform the element-wise product of these two matrices. Finally, the result is passed through a ***Dense size 1*** layer to predict the rating.

General Network Structure:

![](https://i.imgur.com/tQDmHzY.png)
    
###### Model Evaluation

* Simple Matrix Factorization    
    |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
    |-|-|-|-|-|-|
    |train|0.76|0.96|0.71|0.42|0.23|
    |test|0.78|1.01|0.70|0.41|0.22|
    
* Matrix Factorization with GMF

    * ***Glorot-normal***
    
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6155|0.6623|0.8037|0.5212|0.2901|
        |test|0.8411|1.1551|0.6648|0.3793|0.2007|
    
    * ***Glorot-uniform***
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6092|0.6537|0.8058|0.5278|0.2966|
        |test|0.8330|1.1331|0.6708|0.3860|0.2013|
    
    * ***He-normal***
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6220|0.6739|0.7981|0.5164|0.2859|
        |test|0.8287|1.1284|0.6742|0.3889|0.2032|
    
    * ***He-uniform***
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6413|0.7038|0.7870|0.4982|0.2751|
        |test|0.8108|1.0813|0.6820|0.3954|0.2094|
    

***Observations:*** 
On most metrics, GMF outperforms the simple matrix factorization model on the training set, but on the test set, it seems to perform worse. The model is overfitting heavily and still lacks much of the non-linearity we discussed earlier.

We will improve this with the next model.

##### MLP - Multilayer Perceptron

###### Model

The MLP model is a multi-layer neural network. Similar to GMF, our group also creates 2 latent matrices for user and item.

After passing through the 2 latent matrices, we flatten both and concatenate them.

***The main part of the MLP here:***  
Build a neural network with ***num_layer (a list of integers where each number corresponds to the number of hidden nodes for each layer, excluding the first layer, as it is the total dimension output of the 2 embedding layers)*** with ReLU as the activation function.

The final layer is always a ***Dense layer of size 1*** with the predicted rating as the output.

General network structure

![](https://i.imgur.com/EXeF90L.png)

###### Model Evaluation

* Simple Matrix Factorization  
    |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
    |-|-|-|-|-|-|
    |train|0.76|0.96|0.71|0.42|0.23|
    |test|0.78|1.01|0.70|0.41|0.22|

* Matrix Factorization with MLP  

    * ***Glorot-normal***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6497|0.7056|0.7825|0.4851|0.2588|
        |test|0.7337|0.8921|0.7283|0.4377|0.2288|

    * ***Glorot-uniform***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6425|0.7008|0.7905|0.4931|0.2655|
        |test|0.7308|0.8997|0.7345|0.4414|0.2340|

    * ***He-normal***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6091|0.6482|0.8094|0.5232|0.2910|
        |test|0.7216|0.8910|0.7390|0.4514|0.2428|

    * ***He-uniform***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6048|0.6416|0.8122|0.5284|0.2925|
        |test|0.7264|0.9043|0.7380|0.4486|0.2413|

***Comment:*** When using a multi-layer neural network, we immediately see the effectiveness, as all metrics perform better compared to the previously mentioned ***simple Matrix Factorization model***.

Using the neural network introduces non-linearity, and we can observe its effectiveness.

##### Applying Both GMF and MLP    

###### Model
When we apply both GMF and MLP, the model will have the following structure:

![](https://i.imgur.com/1Rub0MR.png)

Since both models are applied together, each network's output will be a matrix. We then concatenate these two matrices and pass them through the final ***Dense size 1 layer*** to predict the rating.

General network structure

![](https://i.imgur.com/pC9Q7P3.png)

###### Model Evaluation
* Simple Matrix Factorization  
    |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
    |-|-|-|-|-|-|
    |train|0.76|0.96|0.71|0.42|0.23|
    |test|0.78|1.01|0.70|0.41|0.22|

* Matrix Factorization with both GMF and MLP  

    * ***Glorot-normal***
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6293|0.6745|0.7955|0.5042|0.2754|
        |test|0.7351|0.8971|0.7297|0.4349|0.2325|

    * ***Glorot-uniform***
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6337|0.6793|0.7924|0.5006|0.2698|
        |test|0.7382|0.9017|0.7248|0.4335|0.2296|

    * ***He-normal***
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6232|0.6645|0.7996|0.5094|0.2790|
        |test|0.7375|0.9035|0.7278|0.4338|0.2286|

    * ***He-uniform***
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6248|0.6715|0.7998|0.5111|0.2788|
        |test|0.7331|0.9029|0.7336|0.4404|0.2344|

***Comment:*** When using both GMF and MLP, we find that although it is more effective than the typical ***Matrix Factorization*** model, the results are nearly identical to the model using only MLP. This shows that to help the model learn the latent features of users and movies, improving the parameters in the NLP model is crucial, as adding the GMF model does not significantly change the results.


***Note:*** We will adjust the dropout rate and provide model evaluation comments in the next section.

##### Calculation of Parameters to Learn
* To learn the embedding layer, we need to learn 640,000 parameters.
* To learn the weight matrix between Flatten and Dense(32), we need to learn (76801 * 32) parameters.
* To learn the weight matrix between Dense(64) and Dense(2), we need to learn (33 * 2) parameters.
***In total, there are 3,097,698 parameters to learn.***

#### Model Evaluation

---

* Test Set

| Rate | Class 0 Accuracy | Class 1 Accuracy | Precision | Recall |
| -------- | -------- | -------- | -------- | -------- |
| 0.85    | 0.68     | 0.88     | 0.82     | 0.68     |
| 0.81    | 0.73     | 0.84     | 0.78     | 0.73     |
| 0.8     | 0.79     | 0.79     | 0.75     | 0.79     |
| 0.795   | 0.75     | 0.82     | 0.77     | 0.75     |
| 0.79    | 0.77     | 0.81     | 0.76     | 0.77     |
| 0.78    | 0.78     | 0.80     | 0.76     | 0.78     |
| 0.77    | 0.76     | 0.81     | 0.76     | 0.76     |
| 0.75    | 0.72     | 0.84     | 0.78     | 0.72     |
| 0.7     | 0.76     | 0.81     | 0.76     | 0.76     |
| 0.65    | 0.73     | 0.83     | 0.77     | 0.73     |
| 0.6     | 0.70     | 0.83     | 0.77     | 0.70     |

***Comments and Evaluation:*** Our team chose a dropout rate of ***rate = 0.8*** because at other rates, as shown in the table above, there is a noticeable ***imbalance*** in accuracy between class 0 and class 1, with class 1 being learned better. Therefore, we decided to select the rate that balances accuracy across both classes 0 and 1 on the test set.

From the table above, we can see that accuracy for both classes is quite balanced when the rate is in the range ***(0.77 - 0.8)***. A notable observation is that as the probability of correctly predicting class 1 increases, the probability of correctly predicting class 0 decreases.

Interestingly, ***higher dropout rates*** seem to be effective. We can explain this as follows:

Just like recognizing a person in a photo from a distant position, where there is a lot of noise ***(e.g., background)*** and few distinguishing features, our problem differs from standard sentiment analysis (where sentences are typically a few dozen words long). Most comments in our dataset, even after preprocessing, still have an average length of ***1000 to 1500*** words. This increases the noise in each sample (since they often discuss the plot, characters they like, dislike, etc.), while the distinguishing features that highlight sentiment are sparse. Therefore, we use more nodes in earlier layers and set the dropout rate high to increase the chances of identifying more features.

## Running the Program Guide
### Data Collection:
Run the ***[2] Data Crawling.py*** file to collect data.

***We suggest you download the data from the link below for testing, as data collection can take a long time.***

    Link: https://drive.google.com/drive/folders/1yNcBF1FhvYpy7XMB-djFmAfBETraAAJf?usp=sharing

The data integration process has been handled during the crawling phase.

### Data Cleaning and Preprocessing
Run the files in the ***[3] Data Cleaning And Data Preprocessing*** folder sequentially.

### Data Exploration
Run the ***[4] Data Exploration.ipynb*** file.

### Experimenting with Models:
+ For basic suggestion models, run the files in the ***modeling/Suggestion/*** folder in the order indicated by the team.

+ For rating prediction models using artificial neural networks, run the ***NeuMF_MLP_GMF.ipynb*** file.

* For the Review Analysis task, run the ***[1] Sentiment Analysis.ipynb*** file.

***Note:*** The results will be stored in the ***result*** folder.
