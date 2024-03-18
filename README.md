# 2019 Indonesian Presidential Election Tweet Sentiment Analysis

This study examines the emotional tone of tweets concerning the 2019 Indonesian Presidential Election. Through employing diverse machine learning methods and models, we investigated the overall sentiment towards the contenders and pinpointed significant trends and subjects within the discussions.

Initially conceived as a collaborative effort (NLP B) during the AI For Indonesia Bootcamp Batch 4, this project involved the contributions of team members such as myself (Satriavi Dananjaya), Roby Koeswojo, Reinaldo Rafael, Rijal Abdulhakim, and Muhammad Yatsrib. Overseeing the project as a mentor was Ferianda Satya.

# Data Understanding
The Dataset shows that the distribution of sentiment class is relatively even between positive, neutral, and negative. So, in this case, the label is relatively balanced and we can use accuracy as the metric.

![image](https://github.com/yatsribplg/NLP-Project/assets/43275881/49465aa1-1443-4f32-a906-c501b58e76f2)


# Data Preparation
The dataset consists of tweets gathered during the period of the 2019 Indonesian Presidential Election. Each tweet is categorized with sentiment labels: positive, neutral, or negative.

1. Data Loading and Exploration: The first phase encompasses loading the tweet data and performing an initial analysis.
2. Data Cleansing: We preprocess the data by eliminating URLs, and hashtags, and standardizing Indonesian colloquial terms.
3. Data Splitting: The dataset is divided into training and testing sets in readiness for model training.

# Exploratory Data Analysis
The projection for all tweets with TF-IDF Vectorizer and Latent Dirichlet Allocation (LDA) is shown in the 3D plot below. We can see that there is no cluster formed from the plot.

![image](https://github.com/yatsribplg/NLP-Project/assets/43275881/473a442e-e309-4871-a86e-e7cc04a29796)

The majority of the words recorded in the dataset are shown in the picture below, we can conclude that the most favorite words are related to economy and politics, like "harga" (price), "ekonomi" (economy), and "Jokowi" (The candidate of the president of Indonesia in 2019 election).

![image](https://github.com/yatsribplg/NLP-Project/assets/43275881/2c02ba37-84c3-49db-a7ff-0697c388a4ac)

# Text Processing

![image](https://github.com/yatsribplg/NLP-Project/assets/43275881/d47ced86-a1d4-4677-906e-40e8f14be46f)

Text Cleaning Result:

![image](https://github.com/yatsribplg/NLP-Project/assets/43275881/497d8a28-321c-408f-9eea-772d197a484a)

The result is not entirely clear, as indicated in the picture above in the red rectangles, definitely some room for improvement.

# Sentiment Analysis Modelling
We performed sentiment analysis employing various methodologies:

Random Forest with TF-IDF: We utilized Random Forest classifiers with TF-IDF vectors derived from the tweets.
Random Forest with Word2Vec: Word2Vec embeddings were employed alongside Random Forest for sentiment classification.
LSTM Networks: LSTM networks were employed using both TF-IDF vectors and Word2Vec embeddings.
Benchmark Models: We evaluated our models against pre-trained models including BERT, Indonesian-roberta-base-sentiment-classifier, NusaX-senti, and GPT-4 Turbo Preview.

# Overall Accuracy

![image](https://github.com/yatsribplg/NLP-Project/assets/43275881/66feebc1-1e0b-4605-ade5-bbf7af6ee38b)

# Best Model

The best result from several modeling scenarios and embeddings performed was achieved using the random forest model and Word2Vec as word embeddings. The training set accuracy was 95.79%, while the validation set accuracy was 61.7%.

Hyperparameters for Random Forest:

N_estimators = 35
Min_sample_split = 3
Min_samples_leaf = 4
Hyperparameters for Word2Vec word embedding:

Min_count = 4
Window = 7
Vector_size = 100

# Benchmark Test Results

# Evaluation
From all the experiments conducted, the best result was obtained using the pre-trained BERT model, with an accuracy on the test set of 63.63%.

![image](https://github.com/yatsribplg/NLP-Project/assets/43275881/498e1238-e6b3-4e94-bd75-c971425af3fc)

The model tends to predict better in negative class, this is because naturally, humans tend to label the sentence to negative class better than positive and neutral.

# Conclusion
1. For the sentiment analysis use case regarding the 2019 election, based on the available dataset, the best outcome was achieved using the random forest model and word2vec word embedding. The accuracy score on the validation set was 61.7%.

2. Out of all the experimented scenarios, overfitting emerged as the primary issue, indicated by the learning curve showing improvements in the training set not reflected in the validation set.

3. Generally, using the Random Forest model is preferable over using the LSTM model.

# Improvement Ideas

1. Enhance the quality of text cleaning results.
2. Prioritize the use of machine learning models over deep learning.
3. Address overfitting issues, for example, through hyperparameter tuning, dropout techniques, or increasing the amount of data (if feasible).
4. Improve data annotation (labeling), focusing on the classes that experience the most misclassification based on the confusion matrix.










