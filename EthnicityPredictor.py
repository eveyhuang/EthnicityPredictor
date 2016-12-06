
import numpy as np
import pandas as pd
import random
import nltk
from nltk import NaiveBayesClassifier,classify
from nltk import ConfusionMatrix
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
import pickle

class ethnicityPredictor():

    ## colName: the name of the column for names (Lastname, Firstname)

    ## ethCol: the name of the column that contains ethnicity information.

    ## ethnicityCol: the name of dummy variable column that indicates the interested etnnicity.

    ## Read in data with specified variables.
    def __init__(self, fileName, colName, ethnicityCol):
        print ("Reading data from file: " + fileName)
        self.df = pd.read_csv(fileName, encoding='latin-1')
        self.colName = colName
        self.ethnicityCol = ethnicityCol
        #self.createDummy(ethCol,ethnicity)

    ## Create a dummy variable to indicate whether each row is the ethnicity we are interested in.
    def createDummy(self,ethCol, ethnicity):
        self.df['is'+ethnicity] = self.df[ethCol].str.contains(ethnicity, na=False).astype(int)
        print ("Created a dummy variable column for " + ethnicity)
        print (self.df.describe())


    def getFeatures(self):

        featureset = list()

        for Name, Japanese in zip(self.df[self.colName], self.df[self.ethnicityCol]):
            features = self.nameFeatures(Name)
            featureset.append((features, Japanese))
        return featureset


    def nameFeatures(self,name):
        name = str(name).upper()
        return {
            'last_two': name[-2:],
            'last_three' : name[-3:],
            'last_is_vowel' : (name[-1] in 'aeiou')
        }

    def trainAndTest(self,trainingPercent=0.80):
        featureset = self.getFeatures()
        print ("Created a set of features.")
        random.shuffle(featureset)
        name_count = len(featureset)
        cut_point=int(name_count*trainingPercent)
        self.train_set = featureset[:cut_point]
        self.test_set  = featureset[cut_point:]
        print ("Created training set and testing set.")

        self.train(self.train_set)

        return self.test(self.test_set)


    def classify(self,name):
        feats=self.nameFeatures(name)
        return self.classifier.classify(feats)


    def train(self,train_set):
        self.classifier = NaiveBayesClassifier.train(self.train_set)
        save_classifier = open("naivebayes.pickle","wb")
        pickle.dump(self.classifier, save_classifier)
        save_classifier.close()
        print("Saved the classifier as naivebayes.pickle")
        return self.classifier

    def getMostInformativeFeatures(self,n=5):
        return self.classifier.most_informative_features(n)

    def test(self,test_set):
        print("NLTK NB accuracy percent:", classify.accuracy(self.classifier,self.test_set))

    
