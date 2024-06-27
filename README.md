PROJECT TITLE: MOVIE GENRE PREDICTION

AIM : Predict the genre of a movie based on its plot summary and other features.

DESCRIPTION : Use natural language processing (NLP) techniques for text classification on a movie dataset.

DATASET: The dataset used for this project consists of 4 columns, 'id', 'movie_name', 'synopsis', and 'genre'.The dataset used can be found in Kaggle.

PROJECT OVERVIEW
This project utilizes Natural Language Processing (NLP) techniques to perform genre prediction based on the plot summary of movies. The goal is to classify movies into their respective genres using machine learning algorithms.

STEPS TAKEN

1. IMPORT LIBRARIES :
Essential libraries such as NLTK, Pandas, Scikit-learn, Matplotlib, and Seaborn are imported for data processing, machine learning, and visualization.

2. LOAD & PREPROCESS DATA :
Load the movie dataset. Perform text preprocessing tasks such as:
Removing punctuation and special characters,
Converting text to lowercase and 
Removing stopwords
3. FEATURE  :
Use TF-IDF vectorization to convert plot summaries into numerical features suitable for classification.
4. MODEL TRAINING :
Split the dataset into training and validation sets.
Train multiple classifiers including:
Multinomial Naive Bayes
Random Forest
Support Vector Machine (SVM)
XGBoost
Evaluate models to select the best-performing classifier.
5. MODEL EVALUATION :
Use accuracy, precision, recall, and F1-score to evaluate model performance.
Analyze the classification report for detailed evaluation.
6. PREDICTING ON TEST DATA :
Transform the test data using the TF-IDF vectorizer.
Predict genres for the test set using the selected classifier.
Evaluate the test set predictions.

TECHNOLOGIES USED : Python, Pandas, NLTK (Natural Language Toolkit), Scikit-learn (sklearn), Matplotlib, Seaborn

CONCLUSION :
This project demonstrates the application of NLP techniques for text classification and supervised learning for genre prediction based on movie plot summaries. By preprocessing text data, extracting features using TF-IDF vectorization, and applying various machine learning algorithms, the project aims to accurately classify movies into their respective genres. The use of multiple classifiers allows for a comprehensive evaluation and selection of the best-performing model, providing insights into the effectiveness of different machine learning approaches for genre prediction.


