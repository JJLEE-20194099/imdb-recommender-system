# Học máy và khai phá dữ liệu

***Chú ý:*** Do có viết 1 số công thức nên nhóm tôi khuyên bạn nên dùng file sau để xem đầy đủ chính xác và chi tiết hơn:
https://hackmd.io/1GFmNyJIT4C4ecAF6Czgcw?both


- [Học máy và khai phá dữ liệu](#)
  * [Thành viên của nhóm](#)
  * [Phân công công việc](#)
- [Giới thiệu đề tài](#)
- [Thu thập dữ liệu](#)
- [Tích hợp dữ liệu](#)
- [Làm sạch và tiền xử lý dữ liệu](#)
- [IMDB datasets](#)
- [Khám phá dữ liệu](#)
- [Một số khái niệm và độ đo cần thiết cho bài toán](#)
- [Mô hình](#)
 
  * [Lọc cộng tác collaborative filtering](#)
      + [Similarity function](#)
          + [Chuẩn hóa dữ liệu](#)
          + [Hàm cosin tính độ tương đồng](#)
    + [Mô hình](#)
        + [User-Item](#)
        + [Item-user](#)
    + [Đánh giá mô hình](#)
      + [MSE](#)
      + [MAE](#)
      + [SIA 1](#)
      + [SIA 0.5](#)
      + [SIA 0.25](#)
     
   * [Phân rã ma trận matrix factorization](#)
      + [Phân tích ma trận thành nhân tử](#)
          + [Ma trận người dùng tiềm ẩn và film tiềm ẩn](#)
          + [Ý nghĩa của các ma trận](#)
     + [Hàm mất mát](#)
     + [Đánh giá mô hình](#)
          + [MSE](#)
          + [MAE](#)
          + [SIA 1](#)
          + [SIA 0.5](#)
          + [SIA 0.25](#)   
    * [Content Based](#)
      + [Xây dựng feature vector bằng TF-IDF](#)
      + [Xây dựng mô hình cho mỗi user](#)
      + [Đánh giá mô hình](#)
          + [MSE](#)
          + [MAE](#)
          + [SIA 1](#)
          + [SIA 0.5](#)
          + [SIA 0.25](#)
    * [Ứng dụng 1 số neural network đơn giản cho bài toán rating prediction](#)
          
       + [Áp dụng MLP và GMF cho phương pháp phân rã ma trận matrix factorization](#)
           + [Ý tưởng](#)
          + [GMF](#)
              + [Mô hình](#)
              + [Đánh giá mô hình](#)
          + [MLP](#)
               + [Mô hình](#)
              + [Đánh giá mô hình](#)
          + [GMF and MLP](#)
               + [Mô hình](#)
              + [Đánh giá mô hình](#)
     * [Phân tích cảm xúc dựa trên đoạn văn review](#)
          + [Tiền xử lý nội dung review](#)
          + [Xây dựng input](#)
        + [Tạo môt mô hình mạng neural đơn giản](#)
        + [Đánh giá mô hình](#)
          + [Confusion Matrix](#)
          + [Precison, Recall](#)

    
    
 
      
- [Hướng dẫn chạy chương trình](#)
  * [Thu thập dữ liệu](#)
  * [Làm sạch và tiền xử lý dữ liệu](#)
  * [Khám phá dữ liệu](#)
  * [Thử nghiệm các mô hình](#)
- [Demo](#)

## Tự đánh giá dựa trên machine learning and data mining project

|Nội dung||
|-|-|
|Thu thập dữ liệu bằng cách sử dụng thư viện BeautifulSoup| Hoàn thành |
| Tích hợp dữ liệu| Hoàn Thành|
| Làm sạch và tiền xử lý dữ liệu| Hoàn thành|
| Phân tích và khám phá dữ liệu các bộ film và rating|Hoàn Thành|
| Phân tích cảm xúc dự đoạn mức độ hài lòng về 1 bộ film trên 2 mức 1-2-3 sao và 4-5 sao dựa vào 1 đoạn văn review |Hoàn thiện|
| Xây dựng tập trung 3 mô hình lọc cộng tác, content based và phân rã ma trận|Hoàn thành|

**Kết quả:** Nhóm đã có những góc nhìn 1 cách tổng quan về các phương pháp dự đoán rating và từ đó liên hệ qua bài toán gợi ý phim cho người dùng


## Thành viên nhóm

|Họ & tên|MSSV|
|-|-|
|Lê Thành Long| 20194099
|Phạm Thế Nam | 20190058
|Đỗ Mạnh Quân| 20194143


***This is a project about how our group built a film recommender system on imdb website. It is a machine learning and data mining project in my university***

***My system's name is ImdbFilm_Recommender***

***@Author: LNQ GROUP***

---

## Phân công công việc trong nhóm

|Họ & tên|Nội dung công việc|
|-|-|
|Lê Thành Long| Code phần crawl data từ trang https://www.imdb.com/ và phương pháp lọc cộng tác collaborative filtering, một số mô hình mạng neural cải thiện kết quả
|Phạm Thế Nam | Tìm hiểu và cài đặt phương pháp phân rã ma trận matrix factorization, slide để thuyết trình
|Đỗ Mạnh Quân| Tìm hiểu và cài đặt phương pháp gợi ý content-based, code mẫu giao diện để demo
|Cả nhóm| Tiền xử lý, tích hợp, khai phá, khám phá và trực quan hóa dữ liệu, và đóng góp ý kiến hoàn chỉnh code, báo cáo và slide.

## Giới thiệu đề tài:
Kéo người dùng về hệ thống của mình là điều mà bất cứ một hệ thống nào hiên nay đều thực sự cần. Khi đó thì hệ thống phải khiến cho người dùng của mình dễ chịu, dễ chịu đặc biệt là cảm thấy tiện lợi khi sử dụng. Vì vậy nhóm chúng tôi đã tìm hiểu vấn đề dự đoán rating và gợi ý các bộ phim cho người dùng mà sao cho người dùng đó khả năng cao là thích, điều này thì nó tối ưu hóa thời gian tìm kiếm phim và tối ưu hóa mức độ tiện lợi khi sử dụng hệ thống.

Nhóm tập trung 3 phương pháp cơ bản chính như collaborative filtering, matrix factorization và content-based.

Đặc biệt, nhóm chúng tôi còn áp dụng thêm GMF, MLP cho phương pháp matrix factorization(sẽ giới thiệu phần cuối project).

Để đánh giá hiệu năng của các mô hình nhóm đã sử dụng phong phú các độ đo và tìm ra độ đo mà nó mang đúng ý nghĩa nhất đối với các mô hình dự đoạn rating từ đó gợi ý cho người dùng.

## Thu thập dữ liệu

Nhóm chúng tôi đã thu thập dữ liệu từ trang review phim nổi tiếng https://www.imdb.com/ mặc dù trên mạng đã có rất nhiều bộ data về cái này tuy nhiên bộ datasets của nhóm chúng tôi rất dồi dào về các thông tin chi tiết phim và thông tin rating và đặc biệt các bộ phim trên bộ dataset của nhóm chúng tôi toàn là phim mới đồng nghĩa là các review ở các bộ phim này cũng là gần đây.

***Thư viện để thu thập dữ liệu là BeautifulSoup***
(Code thu thập dữ liệu ở trong mục src của project)

## Tích hợp dữ liệu

Khi crawl về thì sẽ có khoảng 14 thể loại phim chính trong đó có các id của từng loại

Tích hợp các file id lại với nhau và tiếp tục thu thập dữ liệu về phim và rating

## Làm sạch và tiền xử lý dữ liệu

Quá trình thu thập không tránh khỏi sai sót và lỗi đặc biệt trong quá trình thu thập data thì imdb có cơ chế chống crawl hàng loạt nên bộ dataset có 1 số missing value nhất định:

***Một số bước cơ bản của làm sạch và tiền xử lý dữ liệu:***

+ Đối với các bộ phim có title bằng nan thì xóa luôn
+ Những user nào mà rating 1 bộ phim hơn 2 lần, xóa chỉ giữ lại 1 rating duy nhất
+ Xóa những người nào rating quá nhiều và rating quá ít
+ ...


## IMDB datasets

Gồm 3 bộ chính movie/ml_detail.csv , rating/ml_detail.csv và user/ids với ý nghĩa của các bộ dữ liệu lần lượt là thông tin chi tiết của bộ film, thông tin review về bộ phim đó và các id của user

**Tổng quan và mô tả dữ liệu được sử dụng:**

- **movie/ml_detail.csv:**  8352 dòng và 45 cột

|    | Column             | Dtype   | Value range   | Description                                                                                                            |
|---:|:-------------------|:--------|:--------------|:-----------------------------------------------------------------------------------------------------------------------|
|  0 | movie id                 | string   | nan          | id của bộ film
|  1 | title               | string  | nan           | Tên của bộ phim                                                                                                       |
|  2 | series         | string |nan        | Có phải là series film hay không hay và film lẻ                                                                                         |
|  3 | release year          | int64   | >= 0          | Năm phát hành bô phim                                                             |
|  4 | certification             | string | nan          | Tagger cho bộ phim, phim được phụ huynh cho phép, phim người lớn, ...                                     |
|  5 | duration              | string | nan           | Thời lượng của bộ phim                                                                                                 |                                                  |
|  7 | average rating           | float   | >= 0          | số rating trung bình                                                                                     |
|  8 |rating total             | string   | nan          | tổng số lượt rating                                                                                      |
|  9 | genre list             | string   | nan          | Danh sách các thể loại film                                  |
| 10 | content             | string   | nan          | Content của bộ phim                                  |
| 11 | ...             | ...   | ...          | Bạn đọc tham khảo trong dataset                                                |



- **rating/ml_detail.csv:** 93246 dòng, 9 cột


|    | Column             | Dtype   | Value range   | Description                                                                                                            |
|---:|:-------------------|:--------|:--------------|:-----------------------------------------------------------------------------------------------------------------------|
|  0 | movie id                 | string   | nan          | id của bộ phim được đánh giá                 |
|  1 | name               | string  | nan           | Tên của bộ phim                                                                                                       |
|  2 | user id         | string |    nan         | id của user bình luận                                       |
|  3 | rating          | int64   | 1-5          | Điểm đánh giá bộ phim                                    |
|  4 | content             | string | nan          | Nội dung review bộ phim đó                                                                                              |
|  5 | date              | string | nan           | Thời gian người dùng đánh giá                                                                                                 |                                                  |
|  7 | user index           | int64   | 0-1389          | index của người dùng sau khi tiền xử lý dữ liệu và chuẩn hóa  user id                                      |
|  8 | movie index             | int64   | 0-8351          | index của bộ phim  sau khi tiền xử lý dữ liệu và chuẩn hóa  movie id                                                                                             |


**Chú ý:** Nhóm đã chia data của rating/ml_detail.csv theo ***k-flod (k == 5)*** rồi tính trung bình trên cả 5 fold.

                                      
## Khám phá dữ liệu:

Nhóm chỉ nêu ra 1 số khám phá, chi tiết các bạn đọc ở trong project file ***[4] Data Exploration.ipynb***

+ Tỷ lệ người xem phim hài và tình cảm là cao nhất, tỷ lệ trung bình rating cũng cao 
+ Độ dài của review lúc chưa tiền xử lý trung bình khoảng 1500
+ Phần lớn rating tập trung vào khoảng 3-4-5
+ ...

## Một số khái niệm và độ đo cần thiết trong project
### SIA - Soft Interval Accuracy
Một dự đoán được gói là đúng nói chênh lệch giữa dự đoán và thực tế <= siêu tham số epsilon:
$|y_i - \hat{y_i}| <= eps$
Và ***SIA*** sẽ bằng số mẫu đúng trên toàn bộ mẫu

---

### MAE - Mean Absolute Error

$MAE(y, \hat{y}) = \frac{1}{m} \sum\limits_{i = 1}^{m}|y_i - \hat{y_i}|$

---

### MSE - Mean Squared Error

$MSE(y, \hat{y}) = \frac{1}{m} \sum\limits_{i = 1}^{m}(y_i - \hat{y_i})^2$

---

### RMSE - Root Mean Squared Error

$RMSE(y, \hat{y}) = \sqrt{\frac{1}{m} \sum\limits_{i = 1}^{m}(y_i - \hat{y_i})^2}$

---

### Confusion Matrix
Đặc biệt sử dụng cho các bài toán phân lớp và đây là 1 ma trận vuông có có size là ***số lớp * số lớp***.

Tại hàng i côt j của ma trận có giá trị thuộc khoản từ ***0-1*** có ý nghĩa là số lượng mẫu ***đáng ra là thuộc lớp i nhưng đã dự đoán thuộc lớp j***

---

### Precision và Recall

Với bài toán phân loại 2 lớp thì:
* Precision là tỷ lệ sô mẫu đoán đúng lớp thứ nhất trên số lượng ta dự đoán thứ 1
* Recall là tỷ lệ số mẫu đoán đúng lớp thứ nhất trên số lượng mẫu thực sự thuộc lớp thứ 1

***Chú ý:***
* Precision cao nghĩa là xác suất tìm được các điểm đúng là cao
* Recall cao nghĩa là xác suất mình bỏ sót các điểm đúng là thấp


## Mô hình
### Phương pháp lọc cộng tác collaborative filtering
Metrics đánh giá mà nhóm sử dụng: ***Mean Absolute Error (MAE)*** hoặc ***Soft Interval Accuracy (SIA)*** hoặc ***Root Mean Square Error(RMSE)***

Các features sử dụng cho việc dự đoán: 

|Feature name | Type | 
|---|---|
|movie index |numerical|
|user index|numerical|
|rating|numerical|

#### Similarity Function
##### Chuẩn hóa dữ liệu

Ma trận user-item: có giá trị của mỗi ô là điểm rating của người dùng với bộ phim, với những bộ phim mà chưa có rating thì gán cho giá trị bằng trung bình cộng của các bộ phim người dùng đó đánh giá.

Chuẩn hóa lại các rating bằng cách là lấy giá trị của mỗi ô trừ đi giá trị trung bình của cột ứng với từng người dùng(Giá trị này chình là trung bình cộng của các bộ phim mà người dùng đo đánh giá)

Khi đó ma trận user-item sẽ chưa các giá trị dương và âm và 0. 0 nghĩa người dùng chưa đánh giá bộ phim đó

##### Hàm cosin tính độ tương đồng

Ứng với mỗi cột ta có 1 vector cho chính 1 người dùng.
Giả sử ta có 2 vector u1 và u2.
Để tính độ tương đồng  của 2 vector này thì ta có 1 công thức đơn giản sau:

$\text{cosine_similarity}(u_1, u_2) = cos(u_1,u_2) = \frac{u_1^Tu_2}{||u_1||_2.||u_2||_2}$   **(1)**


Giá trị của hàm consin-simlarity thuộc khoảng [-1, 1]

* Giá trị 1: Hành vi của 2 user hoàn toàn giống nhau
* Giá trị -1: Hành vi của 2 user hoàn toàn khác nhau

#### Mô hình

***Chú ý***: Nhóm chỉ giải thích mô hình user-user còn item-item sẽ tương tự

Sau khi tính được độ tương đồng giữa các user thì ta có 1 ma trận vuông với số hàng và số cột đều bằng số lượng người dùng với  giá trị nơi đường chéo chính là 1.

Khi đó dựa trên ý tưởng của KNN, ta sẽ lấy ra k users gần với user đang xét và tính toán rating của người dùng đối với 1 sản phẩm mà người dùng đó chưa bao giờ đánh giá.

Công thức: 


![](https://i.imgur.com/jDzadkS.png)




***Trong đó:*** N(u, i) là tập k user có độ tương đồng cao nhất với user u , đặc biệt các user này ***đã đánh giá bộ phim i*** 

***VD***:

Để tính điểm rating của **user u** lên **bộ phim i**, không mất tổng quát giả sử:

* Ta cần tính rating của ${u_0}$ lên ${i_0}$
* Xét các người dùng còn lại mà cũng đánh giá bộ phim ${i_0}$ là ${u_1}$, ${u_2}$, ${u_4}$, ${u_5}$
* Ta tính similarities của ${u_0}$ với các người dùng ${u_1}$, ${u_2}$, ${u_4}$, ${u_5}$ theo công thức **(1)**, giả sử là: **0.23, 0.45, -0.56, -0.1**
* Chọn k là số hàng xóm, giả sử ***k = 3***
* Ta sắp xếp và lấy ra 3 user gần với ${u_0}$ nhất là: ${u_1}$, ${u_2}$, ${u_5}$
* Lấy trong bảng các giá trị rating sau khi normalized của các user ${u_1}$, ${u_2}$, ${u_5}$ lên bộ phim ${i_0}$, giả sử là: **0.25, 0.75, 1.25**

* Khi đó kết quả rating của ${u_0}$ lên ${i_0}$ sẽ tính theo công thức **(2)**: 

   $\hat{y}_{i_0,u_0} =  \frac{\text{0.23 x 0.25 + 0.45 x 0.75 + (-0.1) x 1.25}}{\text{|0.23| + |0.45| + |-0.1|}} = 0.3461$




Nhóm chọn ***k = 50*** với bộ dataset đã crawl được(Nhóm đã thử trên nhiều giá trị k khác nhau nhưng mà với giá trị k = 50 thì mô hình tốt trên hầu hết các độ đo)


#### Đánh giá mô hình

Kết quả:


|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.44|0.28|0.98|0.59|0.31|
|test|0.77|0.99|0.72|0.42|0.23|

    
***Nhận xét***: ta thấy mô hình đạt được độ chính xác khá tốt. Hiệu năng trên tập `val` và `test` lệch nhau khá ít ở các độ đo MAE và MSE. Đối với SIA metric thì ta thấy 72% mẫu test mình dự đoán trong vùng chấp nhận được với sai số là 1 và tiếp tục 2 độ đo SIA 0.5 và 0.25 khá thấp trên tập test.
Nhóm dự đoán có thể do trong quá trình review phim thì rất là khó để chúng ta phân biêt được khi nào nên rating 2, khi nào nên rating 3, nên nhóm quyết định sẽ lấy độ đo SIA 1 làm độ đo chính cho các mô hình phía sau và bên cạnh đó có 2 độ đo giúp ta giám sát nữa là MAE tính trung bình trị tuyệt đối độ chênh lệch và MSE trung bình phương độ chênh lệch.


---

### Phương pháp phân rã ma trận matrix factorization
#### Phân tích ma trận thành nhân tử

##### Xấp xỉ bằng 2 ma trận tiềm ẩn

***Ý tưởng*** :Với ma trận user-item và giá trị của mỗi ô là điểm rating mà user đánh giá cho item, trong đó có các ô trông vì có những bộ phim mà người dùng chưa xem. Ta sẽ xấp xỉ ma trận kết quả bằng 2 ma trận ẩn gọi là ma trận người dùng ẩn và ma trận phim ẩn

Gọi ma trận R là ma trận user-item chúng ta thành lập được khi đó ta sẽ xấp xỉ ma trận này bằng tích 2 ma trận ẩn P và Q.

Ma trận R có kích thước m x n với m người dùng và n bộ phim.
Gọi 2 ma trận ẩn là ma trận P có kích thước m x k và ma trận Q có kích thước n x k (lấy k << m, n)

Khi đó:

\begin{equation}
\hat{R}  = PQ^T
\end{equation}


Củ thể đối với điểm rating người dùng u đánh giá bộ phim i ta có:
\begin{equation}
\hat{R_\text{ui}}  = p_u q_i^T
\end{equation}





##### Ý nghĩa của 2 ma trận tiềm ẩn

Ma trận tiềm ẩn P của người dùng trong đó hàng pu của ma trận sẽ chứa các giá trị là mức dộ yêu thích của người dùng u với các đặc trưng của 1 bộ phim ví dụ như thể loại, nội dung, ...

Ma trận tiềm Q của các bộ phim trong đó hàng qi sẽ chứa các giá trị đo lường mức độ sở hữu các đặc trưng như  là tiêu đề, thể loại hay là nội dung bộ phim

***Chú ý***: Ở các công thức trên có đề cập thì ta có thêm bias bu.
Vì sao lại phải thêm bias cho từng user?

Tùy thuộc vào tính cách xem phim của mỗi người mà họ có thể quyết định số điểm mà họ đánh giá cho bộ phim đó.

***Lý do:*** Có người dễ tính, có người khó tính, dễ tính phim hay có thể cho 5 sao nhưng mà khó tính có thể chỉ cho 4 sao hoặc thậm chí là 3 sao chẳng hạn.

Vậy để giải quyết vấn đề này nhóm chúng tôi đã lấy bias của mỗi user là trung bình cộng số điểm mà người đó rating cho những bộ phim mà người ấy xem. Nếu chọn 2.5 thì giả sử người dùng mà khó tính thì giá trị này không đúng cho lắm.




#### Hàm mất mát

Hàm lỗi nhóm chúng tôi sử dụng là trung bình cộng của bình phương độ chênh lệch kết quả dự doán và kết quả thực tế và dùng thêm l2 regularization để tránh overfitting

$\text{RSS}(y, \hat{y}) = {\frac{1}{m} \sum\limits_{i=1}^{m} (y_i - \hat{y}_i)^2}$

$Loss(y, \hat{y}) = \text{RSS}(y, \hat{y}) + \text{Regularization_Func(P, Q)}$

***Chú ý:*** Ở đây nhóm chúng tôi chọn hàm hiệu chỉnh có công thức là:

$\text{Regularization_Func(P, Q)} = \frac{1}{2} * {\lambda} *  (\sqrt{\sum\limits_{i=1}^m \sum\limits_{j=1}^k |p_{ij}|^2} + \sqrt{\sum\limits_{i=1}^n \sum\limits_{j=1}^k |q_{ij}|^2} )$



#### Đánh giá mô hình

Kết quả:
* ***Khi lamda=0***

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|1.46|3.48|0.48|0.23|0.1430|
|test|1.47|3.55|0.48|0.23|0.1438|

* ***Khi lamda=0.1***

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.71|0.42|0.23|
|test|0.78|1.01|0.70|0.41|0.22|


* ***Khi lamda=0.5***

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|


* ***Khi lamda=1***

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.76|0.96|0.72|0.42|0.23|
|test|0.78|1.01|0.71|0.42|0.22|

    
***Nhận xét***: 

Khi ***lamda=0***, ta thấy mô hình đạt độ chính xác khá thấp cả tập train và tập test, đặc biệt tập tain có độ chính xác còn cao hơn tập test. Việc loại bỏ hiệu chỉnh l2 khiến cho mô hình học khá tệ


Khi cho ***lamda != 0*** ta thấy mô hình đạt được độ chính xác khá tốt và tập test và tập train không chênh lệch nhiều Hiệu năng trên tập `val` và `test` lệch nhau khá ít ở các độ đo MAE và MSE. 

Khi ***lamda=0.1, 0.5, 1*** kết quả gần nhữ xấp xỉ nhau nên nhóm chon ***lamda=0.5*** làm kết quả cuối cùng

Đối với SIA metric thì ta thấy 70% mẫu test mình dự đoán trong vùng chấp nhận được với sai số là 1 và tiếp tục 2 độ đo SIA 0.5 và 0.25 khá thấp trên tập test.Mô hình cho kết quả tốt hơn khi thêm hiểu chỉnh l2. Nhóm dự đoán có thể do trong quá trình review phim thì rất là khó để chúng ta phân biêt được khi nào nên rating 2, khi nào nên rating 3, nên nhóm quyết định sẽ lấy độ đo SIA 1 làm độ đo chính cho các mô hình phía sau và bên cạnh đó có 2 độ đo giúp ta giám sát nữa là MAE tính trung bình trị tuyệt đối độ chênh lệch và MSE trung bình phương độ chênh lệch.

***Với lamda=0.5:***
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

---

### Phương pháp gợi ý dựa trên nội dung content-based
#### Xây dựng feature vector bằng TF-IDF

Trong bộ dataset movie/ml_detail.csv, thì chúng ta có 27 cột chi tiết là 27 cột thể loại, 1 bộ phim có thể thuộc nhiều thể loại

***VD:*** Thể loại của 1 bộ phim là ***Animation|Comedy|Family*** khi đó các cột Animation, Comedy và Family của bộ phim có giá trị là 1 và 24 cột thể loại phim còn lại có giá trị 0.

Ta sẽ xây dựng ma trận feature vector bằng TF-IDF trong đó từng hàng của  ma trận chính là feature vector của từng bộ phim

#### Xây dựng mô hình cho mỗi user

Sau khi có được ma trận feature vector, ta sẽ xây dựng mô hình cho riêng từng user, ở đây nhóm chúng tôi dùng mô hình hồi quy đơn giản, củ thể ở đây bọn em dùng các mô hình như hỗ quy tuyến tính (Linear Regression)  và  thêm mô hình hồi quy Ridge (Ridge Regression)

#### Hàm lỗi

$\text{RSS}(y, \hat{y}) = {\frac{1}{m} \sum\limits_{i=1}^{m} (y_i - \hat{y}_i)^2}$

$Loss(y, \hat{y}) = \text{RSS}(y, \hat{y}) + \text{Regularization_Func}$

***Chú ý:*** Ở đây nhóm chúng tôi chọn ***hàm hiệu chỉnh*** có công thức là:

$\text{Regularization_Func} = \frac{1}{2} * {\lambda} *  (\sqrt{\sum\limits_{i=1}^m \sum\limits_{j=1}^k |p_{ij}|^2} + \sqrt{\sum\limits_{i=1}^n \sum\limits_{j=1}^k |q_{ij}|^2} )$

* ***Với hồi quy tuyến tính thì ${\lambda}$ = 0***
* ***Với hồi quy Ridge thì ${\lambda}$ != 0***

***Đầu vào:***

* Đầu vào X của user u là feature extractor của các bộ phim mà u đánh giá
* Đầu vào y của user u là các điểm đánh giá cho các bộ phim mà u đã xem

#### Đánh giá mô hình

Kết quả:

* Mô hình Linear Regression

|Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
|-|-|-|-|-|-|
|train|0.60|0.71|0.78|0.53|0.35|
|test|0.88|1.39|0.67|0.39|0.23|

* Mô hình Ridge Regression:
    + ***(alpha=500)***

        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.76|0.96|0.71|0.42|0.23|
        |test|0.78|1.01|0.70|0.41|0.22|
    

***Nhận xét***: 

Xét trên độ đo chính mà nhóm chúng tôi sử dụng thì trên cả 2 mô hình Ridge và Linear Regression là SIA 1, ta thấy mô hình Linear Regession có 67% mẫu trên tập test được dự đoán thuộc vùng chấp nhận được còn mô hình Ridge với alpha = 500 thì 70% mẫu trên tập test được dự đoán thuộc vùng chấp nhận được.

Sử dụng hồi quy ridge có vẻ hiểu quả hơn hồi quy tuyến tính, sở dĩ là vì có siêu tham số alpha giúp cho hồi quy ridge chống hiện tượng overfitting.

---

### Ứng dụng 1 số neural network đơn giản cho bài toán rating prediction

#### Áp dụng MLP và GMF cho phương pháp phân rã ma trận matrix factorization

##### Ý tưởng

Như các bạn đã biết thì phương pháp phân rã ma trận đã đề cập ở trên thì chúng ta đi tìm 2 ma trận ẩn của người dùng và phim sao cho tích của chúng làm sao để sát với ma trận đánh giá nhất có thể là càng tốt.

Tính đơn giản của phép nhân tích vô hướng của 2 ma trận, rất khó để học được các thuộc tính ẩn của cả user và phim.

Vì vậy ở mục này nhóm chúng tôi xin đề cập tới 1 mạng phân rã ma trận nơ ron có tên là NeuMF. Sở dĩ nhóm chúng tôi đề cập tới mạng nơ ron nhân tạo ở đây bởi vì tính linh hoạt và tính phi tuyến của nó.

Ở trong mục này nhóm chúng tôi sẽ đề cập tới 2 mạng con có tên là phân rã ma trận tổng quát (***Generalized Matrix Factorization) hay GMF*** và 1 mạng con nữa là ***MLP***

nhóm chúng tôi sẽ sử dụng từng mạng một và kết hợp chúng lại với nhau và đánh giá kết quả của các mạng này ở mục sau

##### GMF - Generalized Matrix Factorization

###### Mô hình
Mạng con GMF tương tự như cấu trúc ở phương pháp Matrix Factorization nhóm chúng tôi đã đề cập ở trước với 2 ma trận tiềm ẩn user embedding của user và item embedding của film với size lần lượt là ***(num_users + 1) * latent_dim và (num_items + 1) * latent_dim***

***Trong đó:*** num_users và num_items là số lượng user và item, latent_dim là ***số đặc trưng ẩn*** của phim và user mà bạn muốn   model của bạn học.

***Vì sao lại cộng 1 trong num_users và num_items?***

Như chúng ta đã biết thì hệ thống không có số lượt user và item cố định vì vậy khi đâu vào là 1 user hoặc 1 item mới mà không có trong hệ thống trước đó thì user hoặc item đó sẽ mang giá trị là ***<OOV>(Tương tự như 1 từ không tồn tại trong từ điển)***
    
***Chú ý:*** trong quá trình cài đặt thuật toán nhóm chúng tôi có thể thêm ***Embedding regularizer l2*** để có thể tránh overfitting cho cả user và item
    
Sau khi tạo 2 ma trận tiềm ẩn cho user và item thì ta làm dãn 2 ma trận này và thực hiện tính tích 2 ma trận này (element-wise) và cuối cùng cho qua 1 lớp ***Dense size 1*** và đó chính là điểm rating được predict.
    
Cấu trúc mạng tổng quát:
    
![](https://i.imgur.com/tQDmHzY.png)
    
###### Đánh giá mô hình

* Matrix Factorization đơn giản    
    |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
    |-|-|-|-|-|-|
    |train|0.76|0.96|0.71|0.42|0.23|
    |test|0.78|1.01|0.70|0.41|0.22|
    
* Matrix Factorization khi áp dụng thêm GMF
    
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
        |train|0.6413|0.7038    |0.7870|0.4982|0.2751|
        |test|0.8108|1.0813|0.6820|0.3954|0.2094|
    
    

***Nhận xét:*** 
Hầu hết trên toàn bộ độ đo trên tập train, GMF tốt hơn mô hình matrix factorization, nhưng tập test thì có vẻ là tệ hơn. Mô hình đang bị overfit quá nặng và nhìn qua mô hình là vẫn chưa có thêm nhiều tình phi tuyến như ta đã nói từ trước.
    
Ta sẽ cải thiện với mô hình dưới.    

##### MLP - Multilayer perceptron

###### Mô hình

Mô hình MLP là 1 mạng neural network đa tầng, cũng như GMF nhóm chúng tôi cũng tạo ra 2 ma trận tiềm cho user và item.
    
Sau khi qua 2 ma trận tiềm ẩn thì flatten 2 ma trận này và concatenate chúng lại với nhau.
    
***Phần chính của MLP ở đây:***
Xây dụng 1 neural network có ***num_layer(là 1 list các số nguyên mà mỗi số ứng với số node ẩn của từng tầng trừ tâng đàu tiên ra vì đó là tống số chiều đầu ra của 2 lớp embedding ẩn )*** với hàm kích hoạt là ***relu***

Và lớp cuối cùng chắc chắn là lớp ***Dense size 1*** có giá trị đầu ra là rating được dự đoán
    
Cấu trúc mạng tổng quát

![](https://i.imgur.com/EXeF90L.png)

###### Đánh giá mô hình
* Matrix Factorization đơn giản    
    |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
    |-|-|-|-|-|-|
    |train|0.76|0.96|0.71|0.42|0.23|
    |test|0.78|1.01|0.70|0.41|0.22|
* Matrix Factorization khi áp dụng thêm MLP
    
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

***Nhận xét:*** Khi sử dụng thêm mạng neural đa tầng ta thấy ngay hiệu quả , tất cả độ đo đều tốt hơn so với mô hình ***Matrix Factorization đơn giản đã đề cập trước đó***.

Khi sử dụng mạng neural ta đã thấy thêm tính phi tuyến nó hiệu quả như thế nào  

##### Áp dụng cả GMF và MLP    

###### Mô hình
Khi ta áp dụng của GMF và MLP mô hình sẽ có cấu trúc như sau:

![](https://i.imgur.com/1Rub0MR.png)

Do áp dụng 2 mô hình lại với nhau nên ở đầu ra của mỗi mạng sẽ có 1 ma trận , khi đó ta tiếp tục concatenate 2 matrix này và cho đi qua lớp ***Dense size 1 cuối cùng*** để đưa ra kết quả rating.

Cấu trúc mạng tổng quát

![](https://i.imgur.com/pC9Q7P3.png)

###### Đánh giá mô hình
* Matrix Factorization đơn giản    
    |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
    |-|-|-|-|-|-|
    |train|0.76|0.96|0.71|0.42|0.23|
    |test|0.78|1.01|0.70|0.41|0.22|
* Matrix Factorization khi áp dụng cả GMF và MLP 
    
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
    
    * ***He-normal***
        |Dataset|MAE|MSE|SIA 1|SIA 0.5|SIA 0.25|
        |-|-|-|-|-|-|
        |train|0.6248|0.6715|0.7998|0.5111|0.2788|
        |test|0.7331|0.9029|0.7336|0.4404|0.2344|



***Nhận xét:*** Khi sử dụng cả GMF và NLP ta thấy mặc dù nó hiểu quả so với mô hình ***Matrix Factorization*** thông thường nhưng mà nó gần như xấp xỉ các độ đo so với mô hình khi chi dùng MLP. Điều này chứng minh để mô hình học tốt các đặc trưng ẩn của người dùng và phim thì mô hình nên cải thiện các tham số ở NLP vì có thêm mạng GMF thì kết quả vẫn không thay đổi nhiều
    
---

### Phân tích cảm xúc dựa trên review

#### Tiền xử lý nội dung

***Các hàm tiền xử lý***:
* remove_html(): xóa các thẻ html trong text nếu có
* lower(): Chuyển thành chữ thường các text
* replace_contraction(): thay thế các kiểu viết tắt
    ***VD:***
    * won't -> will not
    * can't -> cannot
    ...
* remove_punctuation(): xóa các dấu câu (?, ., !)
* correct_spellings(): chỉnh lại cho đúng chính tả 
* negation_preprocess(): thay thế các cụm từ có not thành từ phủ định
     ***VD:***
    * not able -> unable
* remove_stopwords(): loại bỏ các từ thừa
* remove_freqwords(): loại bỏ các từ có tần suất xuất hiện nhiều và các từ không màng ý nghĩa cho lắm như (film, movie, imdb)
* remove_rarewords(): Loại bỏ các từ hiếm (nhóm chúng tôi chọn những từ xuất hiện 1 lần là hiếm)
* lemmatize_words(): chuản hóa các từ về dạng nguyên thể
     ***VD:***
    * sent -> send
    * moving -> move
* remove_urls(): xóa các đường dẫn url khỏi nội dung câu review
* remove_numbers(): xóa các con số

#### Xây dựng input:

Đầu tiên ta sẽ phải xây dựng 1 từ điện bằng cách tách ra các từ trong  văn bản, nhóm chúng tôi lựa chọn đồ dài của cuốn từ điện là 10000 từ, những từ nào có tần suất xuất hiện cao thì sẽ đứng đầu từ điển.

Tạo 1 từ OOV (Out Of Vocabulary) là từ luôn đứng đầu từ điển, nó tương ứng với những từ không thuộc từ điển của chúng ta.

Sau khi tạo ra từ điển thì ứng với mỗi từ sẽ có 1 con số xếp hạng ứng với nó, từ nào xuất hiện nhiều sẽ ứng với con số càng bé.

***Chú ý:*** Bắt đầu từ 2 vì OOV luôn ứng với số 1.

Vì các câu review luôn có đọ dài khác nhau nên nhóm quyết định chọn 1 độ dài nhất định cho toàn bộ data (Nhóm chọn max_length = 1200). Đối với những câu ngắn sẽ padding 0 vào sau và những câu dài sẽ cắt phần sau đi.(Sở dĩ chọn 1200 vì hầu hết các đoạn review trong dataset hầu hết có độ dài từ 1000 đến 1500)

Khi đó ta đã xây dựng được đầu vào là 1 ma trận có chiều là m * max_length với m là số câu cần train và max_length = 1200

#### Tạo một mô hình mạng neural đơn giản

Xét 1 mẫu sample là 1 đoạn nội dung review của 1 bộ phim nào đó. Sau khi các bước chuẩn bị input trên thì mẫu sample này sẽ là 1 ma trận có kích thước 1200*1. Khi đó giá trị tại mỗi ô chính là 1 con số đại diện cho vị trí 1 từ trong câu mapping với từ điển chúng ta đã xây dựng từ trước. Khi mã hóa one hot con số này , do từ điển có 10000 từ nên ta sẽ thu được đầu ra là 1 ma trận 1200 * 10000 chỉ chưa 0 và 1.

Ta cho đầu vào qua 1 lớp Embedding có kích thước 10000 * 64, ta sẽ có 1 ma trận 1200 * 64.

Khi đó có nghĩa là sau khi đi qua lớp Embedding thì từ 1 mẫu câu (ma trận 1200 * 1) ta đưa ra 1 ma trận (1200 * 64) khi đó từng hàng của ma trận đầu ra chính là vector word embedding cho cái từ ở vị trị tương ứng với mỗi hàng.

Ta tiếp tục duỗi ma trận 300 * 64 bằng lớp Flatten có kích thước 1 * 76800 và ta sẽ thu nhỏ lại bằng 1 lớp fully connected (Lớp Dense) có kích thước 32 * 1 và sau đó thu nhỏ lại lớp Dense cuối cùng (Lớp đầu ra ) chỉ có kích thức 2 * 1, dạng [a, b] với a là xác suất dự đoán đúng lớp 0 và 1 là xác suất dự đoán đúng lớp 1

Có thế trong quá trình cài đặt nhóm chúng tôi sẽ thêm 1 số lớp Dropout vào, tuy nhiên mạng cơ bản của nhóm có cấu trúc như sau:

```
Input Vector -> Embedding -> Flatten -> Dropout(rate) -> FC(32 units) + ReLU -> FC(2 units) + Softmax
```

***Chú ý:*** Ta sẽ thay đổi rate của lớp dropout và đưa ra nhận xét trong mục đánh giá mô hình

##### Tính toán số lượng tham số cần phải học
* Để học được lớp Embedding thì ta phải học 640000 tham số
* Để học được ma trận trọng số giữa Flatten và Dense(32) ta phải học (76801 * 32 tham số)
* Để học được ma trận trọng số giữa Dense(64) và Dense(2) ta phải học (33 * 2 tham số)
***Tổng cổng có 3.097.698 tham số cần phải học***

#### Đánh giá mô hình

---
    
* Tập test

| Tham số rate| Độ chính xác lớp 0 | Độ chính xác lớp 1 |Precition| Recall
| -------- | -------- | -------- | -------- | --------|
| 0.85    | 0.68     | 0.88     |0.82|0.68
| 0.81     | 0.73     | 0.84     |0.78|0.73
| 0.8     | 0.79     | 0.79     |0.75|0.79
| 0.795    | 0.75     | 0.82     |0.77|0.75
| 0.79     | 0.77     | 0.81     |0.76|0.77
| 0.78     | 0.78     | 0.80     |0.76|0.78
| 0.77     | 0.76     | 0.81     |0.76|0.76
| 0.75     | 0.72     | 0.84     |0.78|0.72
| 0.7    | 0.76     | 0.81     |0.76|0.76
| 0.65     | 0.73     | 0.83     |0.77|0.73
| 0.6     | 0.70     | 0.83     |0.77|0.70
    
***Nhận xét và đánh giá:*** nhóm chúng tôi chọn tham số ***rate=0.8*** của lớp Dropout, sở dĩ ở 1 số rate khác như bảng bên trên thì độ chính xác trên 2 lớp 0 và 1 đang bị ***chênh lệch hơi nhiều*** và củ thể là lớp 1 đang có vể được học tốt hơn. Vì vậy nhóm chúng tôi quyết định chọn rate nào để cho cân bằng độ chính xác trên tập test của mô hình đối với 2 lớp 0 và 1.
    
Theo quan sát ở bảng trên thì khi ta thấy độ chính xác 2 lớp khá cân bằng khi rate thuộc các vùng từ ***(0.77-0.8)*** và 1 điểm đặc biệt là khi mà xác suất dự đoán đúng lớp 1 càng cao thì xác suất đoán đúng lớp 0 càng thấp.
    
Đặc biệt thì ***drop rate*** ở các vùng giá cao có vẻ là hiệu quả. nhóm chúng tôi có thể giải thích vấn đề này như sau:

Cũng như là 1 bức ảnh nhận dạng 1 người ở 1 vị trị không quá gần chẳng hạn thì các noise rất nhiều ***(VD: background)***, trong khi đó các đặc trưng thì lại quá ít, đối chiếu với bài toán của nhóm chúng tôi khác hoàn toàn với 1 bài toán phân tích cảm xúc thông thường (Vì nhiều bộ datasets đó độ dài của 1 câu chi tầm khoảng vài chục từ), hầu hết các câu bình luận trong datasets mặc dù đã tiền xử lý những vẫn có độ dài trung bình khoảng ***1000 đến 1500***. Điều này càng làm cho 1 mẫu trong datasets các nhiễu thì cực nhiều (Chưa đánh giá phim ngay vì họ còn nói qua về nội dung phim, thích nhân vật nào, ghét nhân vật nào chẳng hạn), trong khi đó các đặc trưng để làm nổi bật lên sự thích thú lại quá ít trong 1 câu. Vì thế nên các lớp trước nhóm chúng tôi cho nhiều node lên , rồi để drop rate ở mức cao nhằm tăng xác suất chọn được nhiều đặc trưng 

## Hướng dẫn chạy chương trình
### Thu thập dữ liệu:
Các bạn chạy file ***[2] Data Crawling.py*** để thu thập dữ liệu.
    
***Nhóm suggest bạn đọc nên lấy data ở link dưới đây test vì quá trình thu thập dữ liệu khá lâu và mất thời gian***

    Link: 

Quá trình tích hợp dữ liệu nhóm đã làm trong quá trình crawl

### Làm sạch và tiền xử lý dữ liệu
Chạy lần lượt các file trong thư mục ***[3] Data Cleaning And Data Preprocessing*** 

### Khám phá dữ liệu
Chạy file ***[4] Data Exploration.ipynb***

### Thử nghiệm các mô hình:
+ Đối với các mô hình suggesion cơ bản, bạn đọc chạy lần lượt các file từ trên xuống dưới theo thứ tự mà nhóm mình đã đánh dấu nơi tên file trong thư mục ***modeling/Suggestion/***

+ Đối với các mô hình dự đoán rating dùng mạng nơ ron nhân tạo thì các bạn chạy file ***NeuMF_MLP_GMF.ipynb***

* Đối với bài toán Review Analysis thì các bạn chạy file ***[1] Sentiment Analysis.ipynb***

***Chú ý:*** Kết quả chạy trong thư mục ***result***










