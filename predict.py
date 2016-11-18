import numpy as np
import pandas as pd
import random
from nltk import NaiveBayesClassifier,classify
from nltk import ConfusionMatrix
import pickle


def nameFeatures(name):
    	name = str(name).upper()
    	return {
        	'last_two': name[-2:],
        	'last_three' : name[-3:],
        	'last_is_vowel' : (name[-1] in 'aeiou')
    	}

if __name__ == "__main__":
        classifier_f = open("naivebayes.pickle", "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        name = input('Enter name to classify: ')
        print ('\n%s is classified as %s'%(name, classifier.classify(nameFeatures(name))))