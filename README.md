# Prediction with Amazaon Alexa Reviews

## By Elena Korshakova and Diedre Brown
### Pratt Institute, Fall 2020

This project aims to analyze these reviews and ratings of Amazon Alexa devices from 2017 (as training data) and 2018 (as testing data) to develop an algorithm to predict star ratings based on user reviews. Through the process, we aim to determine if star ratings and an end-to-end sentiment analysis of user review comments from 2017 we can predict with accuracy, the star ratings of 2018. If successful, our findings have the potential to provide tech companies with a summary of user preferences and top criteria for the development of future products, as well as, assist in predicting user reactions to recently implemented features based on data collected from previous reviews of similar features.

The project folders include the following:

- **Original Datasets**
	- These datasets were obtained from [kaggle](https://www.kaggle.com/) and include [Amazon Reviews 2017](https://www.kaggle.com/PromptCloudHQ/amazon-echo-dot-2-reviews-dataset), which became the training set; and, [Amazon Reviews 2018](https://www.kaggle.com/sid321axn/amazon-alexa-reviews), which became the test set.
- **Images**
	- This folder includes charts created during the exploratory data analysis and sentiment analysis of the project. For more information on these charts, please refer to our paper and/or our presentation.
- **Models**
	- These are notebooks containing the models developed during this project:
	* [Logisting Regression + TF-IDF and BERT](rating_prediction_log_reg.ipynb)

	* [Random Forest + TF-IDF and BERT](rating_prediction_ensemble.ipynb)

	* [LSTM Neural Network](rating_prediction_neural_net.ipynb)
	
- **Notebooks**
	- These notebooks include the exploratory data analysis and sentiment analysis.
- **Presentation**
	- These files include our paper and a pdf of our presentation, which was delivered on 8 December 2020. 

To use these files:


- **Results**
We trained our model on the data from 2017 and measured accuracy and F1 score on the data from 2018. 

| Method           | Accuracy      | F1    |
| -----------------|:-------------:| -----:|
| TF-IDF + Log Reg |  0.68         |  0.68 |
| BERT + Log Reg   |  0.49         |  0.56 |
| TF-IDF + RF      |  0.72         |  0.68 |
| BERT +  RF       |  0.65         |  0.59 |
| LSTM             |  0.71         |  0.69 |

TF-IDF + Random Forest and LSTM neural network gave the best scores on the test set. As we can see from the table above, BERT embeddings didn't give a boost in performance for our data. Also, such a simple method as TF-IDF encoding + Logistic Regression gave pretty good scores in comparison to Random Forest and Neural Network. 
