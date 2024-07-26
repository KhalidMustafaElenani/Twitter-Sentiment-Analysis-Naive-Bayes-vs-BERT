# Twitter Sentiment Analysis: Naive Bayes vs BERT
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-%20blue?style=plastic)
![Sentiment_Analysis](https://img.shields.io/badge/Sentiment%20Analysis-%20blue?style=plastic)
![License](https://img.shields.io/badge/license%20-%20MIT%20-%20darkblue?style=plastic)
![Transformers](https://img.shields.io/badge/Transformers-4.33.0-%20blue?style=plastic)
![Tensorflow](https://img.shields.io/badge/Tensorflow-2.13.0-%20blue?style=plastic)
![Numpy](https://img.shields.io/badge/Numpy-1.24.4-%20blue?style=plastic)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-%20blue?style=plastic)
![Scikit_Learn](https://img.shields.io/badge/Scikit_Learn-1.2.2-%20blue?style=plastic)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.2-%20blue?style=plastic)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-%20blue?style=plastic)
![Imbalanced_Learn](https://img.shields.io/badge/Imbalanced_Learn-0.10.1-%20blue?style=plastic)
![Build_Status](https://img.shields.io/badge/build-passing-brightgreen)
![Open_Issues](https://img.shields.io/badge/Issues%20-%200%20-%20orange?style=plastic)

## Introduction
This repository contains a comprehensive approach to sentiment analysis on Twitter data, utilizing both Naive Bayes and BERT models. The project is divided into two stages: data preprocessing and model application. The primary goal is to transform raw Twitter data into a suitable format for deep learning models and then compare the performance of Naive Bayes and BERT on this processed data.

## Table of Contents
1. [Overview](#overview)
2. [Dependencies Installation](#dependencies-installation)
3. [Environment Setup](#environment-setup)
4. [Data Retrieval and Preprocessing](#data-retrieval-and-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Text Cleaning and Preparation](#text-cleaning-and-preparation)
7. [Tokenization and Data Structuring](#tokenization-and-data-structuring)
8. [Sentiment Encoding and Balancing](#sentiment-encoding-and-balancing)
9. [Model Implementation and Comparison](#model-implementation-and-comparison)
10. [Results and Insights](#results-and-insights)
11. [Optimization and Tuning](#optimization-and-tuning)
12. [Conclusion](#conclusion)
13. [License](#license)

## Overview
This project aims to conduct sentiment analysis on a dataset of tweets related to COVID-19. The analysis leverages the power of advanced natural language processing models, BERT and Naive Bayes, to classify the sentiment of tweets as positive, negative, or neutral.

## Dependencies Installation
Before starting, install the necessary dependencies:
```bash
pip install emoji
pip install numpy pandas seaborn matplotlib tensorflow transformers imblearn nltk
```

## Environment Setup
1. Import Libraries
```bash
import os
import re
import nltk
import emoji
import string
import zipfile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import BertTokenizerFast
from imblearn.over_sampling import RandomOverSampler
```
2. Mount Google Drive
```bash
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

## Data Retrieval and Preprocessing
1. Download the dataset from Kaggle using the command: ``` kaggle datasets download -d datatattle/covid-19-nlp-text-classification ```
2. Use the provided scripts to preprocess the data. Ensure that the dataset is placed in the appropriate directory for further processing.

## Dataset
<p align="center">
  <img src="Coronavirus.jpg" alt="Coronavirus" width="500"/>
</p>
- The dataset used for this project is the COVID-19 NLP Text Classification dataset available on Kaggle.
- This dataset contains tweets related to COVID-19, labeled with sentiment categories. You can download it from [this link](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)

## Exploratory Data Analysis
Perform an exploratory data analysis (EDA) to understand the dataset and visualize key patterns and distributions. This includes generating summary statistics and visualizing data distributions.

## Text Cleaning and Preparation
Apply text cleaning procedures such as:
- Removing stop words
- Removing punctuation
- Removing special characters
Prepare the text for analysis by transforming it into a suitable format.

## Tokenization and Data Structuring
1. Tokenize the text data using:
```bash
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize(text)
```
2. Structure the data appropriately for both Naive Bayes and BERT models.

## Sentiment Encoding and Balancing
1. Convert sentiment labels into numerical values for model training.
2. Apply techniques to balance the dataset if necessary:
```bash
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
```

## Model Implementation and Comparison
1. Implement the Naive Bayes model using sklearn:
```bash
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)
model = MultinomialNB()
model.fit(X, y)
```
2. Implement the BERT model using transformers:
```bash
from transformers import TFBertForSequenceClassification, BertTokenizerFast
import tensorflow as tf

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
```
3. Evaluate and compare the performance of both models using appropriate metrics such as accuracy, precision, recall, and F1-score.


## Results and Insights
Summarize the results obtained from both models and provide insights based on the analysis. Highlight key findings and performance differences.

## Optimization and Tuning
Discuss any optimization or tuning performed to improve model performance. Include details on hyperparameter tuning, model adjustments, and performance improvements.

## Conclusion
Provide a conclusion based on the results of the comparison. Suggest any future work or improvements that could be undertaken to enhance the analysis.

## License
- This project is licensed under the MIT License - see the LICENSE file for details.
- Feel free to adjust any details to better fit your specific implementation or project structure.
