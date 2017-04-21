import numpy as np
import pandas as pd
import random
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import scipy.sparse
from sklearn.externals import joblib
import re
import pickle
import dill
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import platform


df = pd.read_csv("5percentdata.csv", encoding='latin-1')

with open("clf.pkl", 'rb') as in_strm:
    classifier = dill.load(in_strm)
featureset=[]
for Lastname, Firstname in zip(df["LName"], df["FName"]):
    fName = str(Firstname).lower()
    lName = str(Lastname).lower()    	
    if len(fName) != 1:
        featureset.append(lName + ' ' + fName)
    else:
		featureset.append(lName)

new_preds = classifier.predict(featureset)
df["prediction"] = new_preds

cm = metrics.confusion_matrix(df["Japanese"], df["prediction"])
print(cm)
sensitivity = float(cm[1][1]) / (cm[1][0]+cm[1][1])
specificity = float(cm[0][0]) / (cm[0][0]+cm[0][1])
weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
print('Sensitivity ',sensitivity, '; Specificity: ', specificity)
print('Weighted Accuracy: ', weightedAccuracy)

df.to_csv("predicted_file.csv", encoding='latin-1')


#ap = predict("5percentdata.csv","SVMclassifier.pickle","FName","LName", "predicted")
#ap.apply()

    	