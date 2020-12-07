# Prediction with Amazaon Alexa Reviews

## By Elena Korshakova and Diedre Brown
### Pratt Institute, Fall 2020
* * *

## Project Aims:
This project aims to analyze the reviews and ratings of Amazon Alexa devices from 2017 (as training data) and 2018 (as testing data) to develop an algorithm to predict star ratings based on user reviews. Through the process, we aim to determine if star ratings and a sentiment analysis of user review comments from 2017 can predict with accuracy, the star ratings of 2018. If successful, our findings have the potential to provide tech companies with a summary of user preferences and top criteria for the development of future products, as well as, assist in predicting user reactions to recently implemented features based on data collected from previous reviews of similar features.

The project folders include the following:

- **[Original Datasets](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/original_datasets)**
	- These datasets were obtained from [kaggle](https://www.kaggle.com/) and include [Amazon Reviews 2017](https://www.kaggle.com/PromptCloudHQ/amazon-echo-dot-2-reviews-dataset), which became the training set; and, [Amazon Reviews 2018](https://www.kaggle.com/sid321axn/amazon-alexa-reviews), which became the test set.
- **[Data](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/data )**
	- This folder contains the cleaned versions of the datasets used during our analysis.
- **[Images](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/img)**
	- This folder includes charts created during the exploratory data analysis and sentiment analysis of the project. For more information on these charts, please refer to our paper and/or our presentation.
- **[Models](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/models)**
	* [Logisting Regression + TF-IDF and BERT](https://github.com/diedrebrown/predictionwithalexareviews/blob/main/models/rating_prediction_log_reg.ipynb)
	* [Random Forest + TF-IDF and BERT](https://github.com/diedrebrown/predictionwithalexareviews/blob/main/models/rating_prediction_ensemble.ipynb)
	* [LSTM Neural Network](https://github.com/diedrebrown/predictionwithalexareviews/blob/main/models/rating_prediction_neural_net.ipynb)
- **[Notebooks](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/notebooks)**
	- Exploratory Data Analysis 
	- Text Sentiment Analysis & Topic Modeling
- **[Presentation](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/presentation)**
	- Here you will find a pdf of our presentation, which was delivered on 8 December 2020. 

## Methods:
Preliminary review of the data revealed that the 2017 dataset contained 3662 verified customer reviews for Amazon Dot2 Echo and the 2018 Amazon Reviews dataset contained 3150 verified customer reviews of Amazon devices. The main features of each dataset included date of review, star rating, customer review, type of device, and feedback on specific device performance. Though the 2018 data includes all Amazon devices, during this preliminary review we found that the reviews consisted of feedback on features common to all Amazon Alexa devices. Therefore, we included all reviews in our initial analysis.

We each began the study by conducting an extensive exploratory data analysis (EDA) and data cleaning of both datasets. After discussing how we cleaned the data, we combined our methods to produce a consistent EDA. During these procedures, null values were removed, all the dates were formatted for consistency between the two datasets, and columns such as configuration, color, review useful, and user verified were removed as these columns were blank or contained data that was not needed for this analysis. The cleaned data were exported as pickle files for use in the models. Based on our review of the data during EDA, we separated the analysis into two sections: sentiment analysis and rating prediction.
- Methods of Sentiment Analysis:
	- Used the spaCy library to normalize the data to make patterns more easily detectable by removing stopwords. The spaCy library was used to do this, as it is the most robust and efficient library for natural language processing. As spaCy can be applied to many languages and has integrations built in for neural network models, it provided eased much of the work we needed.
	- Creation of a document term matrix for use in topic modeling with Latent Dirichlet Allocation (LDA). The LDA was used to finds groups of words (topics) that appear frequently together.
- Rating Star Predictions with Machine Learning Models ***ADD WHAT THESE MODELS DO/REPRESENT***:
	- Tf-IDF Encoding. To 
	- Logistic Regression 
	- Random Forest 
	- Bidirectional Encoder Representations from Transformers (BERT)
	- LSTM Neural Network

## Findings:
We trained our model with the data from the 2017 Alexa Ratings Dataset and measured its accuracy and F1 score using the 2018 Alexa Ratings Datset.

| Method             | Accuracy       | F1      |
| :------------------|:--------------:| -------:|
| TF-IDF + Log Reg   |  0.68          |  0.68   |
| BERT + Log Reg     |  0.49          |  0.56   |
| TF-IDF + RF        |  0.72          |  0.68   |
| BERT +  RF         |  0.65          |  0.59   |
| LSTM               |  0.71          |  0.69   |
| _MultiNB + Log Reg_|  _0.92_        |  _0.53_ |

TF-IDF + Random Forest and LSTM neural network gave the best scores on the test set. As we can see from the table above, BERT embeddings didn't give a boost in performance for our data. Also, such a simple method as TF-IDF encoding + Logistic Regression gave pretty good scores in comparison to Random Forest and Neural Network. 

**ADD FINDINGS FROM SENTIMENT ANALYSIS**

## Study Limitations:
- _Small Dataset & the Balance of the Dataset_ As 2018 market reports revealed, approximately 39 million Americans own an Amazon Echo or Google Home (Perez, 2018). While this is a substantial figure, the amount of publicly data available to analyze these user's experience is limited. Furthermore, the users that tend to leave reviews on consumer sites like Amazon tend to be very positive or negative, which results in imbalanced data.    
- _Time of Year Seasonality_ The 2017 dataset covers Sept 1st - Oct 30th. While the 2018 set covers only the month of July. As we discovered with an outlier review, there was a reduction in price this month due to Amazon Prime Day specials.
- _Paid Reviews?_ It is possible that the data contained paid reviews, which are generally positive. However, as all users were verified, more data was needed to determine the veracity of these reviews. 
- _Multitude of Amazon Devices_ The reviews covered user experiences with a number of Amazon devices. While the reviews consisted of the features that are common to all of these devices, without data on the user's rationale for selection of the device type we cannot state how the device itself contributed to the user's experience with Alexa.

## Discussion:
This study showed that consumer reviews are great for judging initial reactions to new product features. From the reviews, it is possible to confirm "likeability" of new product features and initial reactions and predict the potential changes for future product features; however, the accuracy of these reviews is questionable without a lot of data.

For future work on this study, we recommend the use of a larger dataset. These data could include reviews from 2019 on and/or be obtained via web scraping. We attempted to use more advanced methods of encoding, such as [GloVe](https://nlp.stanford.edu/projects/glove/); however, we both received too many issues within the allowable time to complete this project. As an unsupervised learning algorithm for word represetation, GloVe could have provided stronger linear substructures and more nuance between word pairs to our analysis. Additionally, we think using a recurrent neural network (RNN) for text processing could result in more advanced summarization than topic modeling alone. However, we believe comparable results could also be acheieved with extended experimentations with hyperparameter tuning for logarithmic regression, random forest, and neural network parameters.


## References:
- Perez, S. (2018). 39 million Americans now own a smart speaker, report claims. Retrieved from https://techcrunch.com/2018/01/12/39-million-americans-now-own-a-smart-speaker-report-claims/