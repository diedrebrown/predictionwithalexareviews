# Prediction with Amazaon Alexa Reviews

## By Elena Korshakova and Diedre Brown
### Pratt Institute, Fall 2020
* * *

## Project Aims:
This project aims to analyze the reviews and ratings of Amazon Alexa devices from 2017 (as training data) and 2018 (as testing data) to develop an algorithm to predict star ratings based on user reviews. Through the process, we aim to determine if star ratings and a sentiment analysis of user review comments from 2017 can predict with accuracy, the star ratings of 2018. If successful, our findings have the potential to provide tech companies with a summary of user preferences and top criteria for the development of future products, as well as, assist in predicting user reactions to recently implemented features based on data collected from previous reviews of similar features.

The project folders include the following:

- **[Original Datasets](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/original_datasets)**
	- These datasets were obtained from [kaggle](https://www.kaggle.com/) and include [Amazon Reviews 2017](https://www.kaggle.com/PromptCloudHQ/amazon-echo-dot-2-reviews-dataset), which became the training set; and, [Amazon Reviews 2018](https://www.kaggle.com/sid321axn/amazon-alexa-reviews), which became the test set.
- **[Data](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/data**)**
	-This folder contains the cleaned versions of the datasets used during our analysis.
- **[Images](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/img)**
	- This folder includes charts created during the exploratory data analysis and sentiment analysis of the project. For more information on these charts, please refer to our paper and/or our presentation.
- **[Models](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/models)**
	* [Logisting Regression + TF-IDF and BERT](https://github.com/diedrebrown/predictionwithalexareviews/blob/main/models/rating_prediction_log_reg.ipynb)
	* [Random Forest + TF-IDF and BERT](https://github.com/diedrebrown/predictionwithalexareviews/blob/main/models/rating_prediction_ensemble.ipynb)
	* [LSTM Neural Network](https://github.com/diedrebrown/predictionwithalexareviews/blob/main/models/rating_prediction_neural_net.ipynb)
- **[Notebooks](https://github.com/diedrebrown/predictionwithalexareviews/tree/main/notebooks)**
	- Exploratory Data Analysis 
	- Text Sentiment Analysis & Topic Modeling
- **[Presentation]()**
	- Here you will find a pdf of our presentation, which was delivered on 8 December 2020. 

## Methods:
Preliminary review of the data revealed that the 2017 dataset contained 3662 verified customer reviews for Amazon Dot2 Echo and the 2018 Amazon Reviews dataset contained 3150 verified customer reviews of Amazon devices. The main features of each dataset included date of review, star rating, customer review, type of device, and feedback on specific device performance. Though the 2018 data includes all Amazon devices, during this preliminary review we found that the reviews consisted of feedback on features common to all Amazon Alexa devices. Therefore, we included all reviews in our initial analysis.

We each began the study by conducting an extensive exploratory data analysis (EDA) and data cleaning of both datasets. After discussing how we cleaned the data, we combined our methods to produce a consistent EDA. During these procedures, null values were removed, all the dates were formatted for consistency between the two datasets, and columns such as configuration, color, review useful, and user verified were removed as these columns were blank or contained data that was not needed for this analysis. The cleaned data were exported as pickle files for use in the models. Based on our review of the data during EDA, we separated the analysis into two sections: sentiment analysis and rating prediction.
- Methods of Sentiment Analysis:
	- Used the spaCy library to normalize the data to make patterns more easily detectable by removing stopwords. The spaCy library was used to do this, as it is the most robust and efficient library for natural language processing. As spaCy can be applied to many languages and has integrations built in for neural network models, it provided eased much of the work we needed.
	- Creation of a document term matrix for use in topic modeling with Latent Dirichlet Allocation (LDA). The LDA was used to finds groups of words (topics) that appear frequently together.

## Findings:
We trained our model with the data from the 2017 Alexa Ratings Dataset and measured its accuracy and F1 score using the 2018 Alexa Ratings Datset.

| Method             | Accuracy      | F1    |
| :------------------|:-------------:| -----:|
| TF-IDF + Log Reg   |  0.68         |  0.68 |
| BERT + Log Reg     |  0.49         |  0.56 |
| TF-IDF + RF        |  0.72         |  0.68 |
| BERT +  RF         |  0.65         |  0.59 |
| LSTM               |  0.71         |  0.69 |
| _MultiNB + Log Reg_|  0.92         |  0.53 |

TF-IDF + Random Forest and LSTM neural network gave the best scores on the test set. As we can see from the table above, BERT embeddings didn't give a boost in performance for our data. Also, such a simple method as TF-IDF encoding + Logistic Regression gave pretty good scores in comparison to Random Forest and Neural Network. 



## Study Limitations:


## Discussion:
