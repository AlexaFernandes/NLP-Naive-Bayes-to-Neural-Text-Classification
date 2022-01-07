# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 01:32:36 2021

@author: alexa
"""
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics

sns.set() # use seaborn plotting style

# Load the dataset
data = fetch_20newsgroups()
# Get the text categories
text_categories = data.target_names
# define the training set
train_data = fetch_20newsgroups(subset="train", categories=text_categories)
# define the test set
test_data = fetch_20newsgroups(subset="test", categories=text_categories)

print("We have {} unique classes".format(len(text_categories)))
print("We have {} training samples".format(len(train_data.data)))
print("We have {} test samples".format(len(test_data.data)))

# let’s have a look as some training data
#print(test_data.data[5])

print("1")
# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
print("2")
# Train the model using the training data
model.fit(train_data.data, train_data.target)
print("3")
# Predict the categories of the test data
predicted_categories = model.predict(test_data.data)
print("4")

print(np.array(test_data.target_names)[predicted_categories])

# plot the confusion matrix
mat = confusion_matrix(test_data.target, predicted_categories)
fig = plt.figure(num=None, figsize=(10, 10), dpi=80)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=train_data.target_names,yticklabels=train_data.target_names)

plt.xlabel("true labels")
plt.ylabel("predicted label")

plt.savefig('NB_graph.png', bbox_inches='tight')
plt.show()

print(metrics.classification_report(test_data.target, predicted_categories,target_names=text_categories))

print("The accuracy is {}".format(accuracy_score(test_data.target, predicted_categories)))

# custom function to have fun
def my_predictions(my_sentence, model):
    all_categories_names = np.array(data.target_names)
    prediction = model.predict([my_sentence])
    return all_categories_names[prediction]

# my_sentence = "jesus"
# print(my_predictions(my_sentence, model))

