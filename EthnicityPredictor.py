
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
import dill
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import platform



class ethPredictor():

    ## colName: the name of the column for names (Lastname, Firstname)

    ## ethCol: the name of the column that contains ethnicity information.

    ## ethnicityCol: the name of dummy variable column that indicates the interested etnnicity.

    ## Read in data with specified variables.

    def __init__(self, fileName, lastName, ethnicityCol,n_components, class_weight, lossfn, penalty, transformer, modelName,firstName='FName'):
        print ("Reading data from file: " + fileName)
        self.df = pd.read_csv(fileName, encoding='latin-1')
        self.lastName = lastName
        self.firstName = firstName
        self.ethnicityCol = ethnicityCol
        self.ratio = self.df.groupby(ethnicityCol).count()
        print(self.ratio)
        self.n_components = n_components
        self.class_weight=class_weight
        self.lossfn =lossfn
        self.penalty=penalty
        self.transformer = transformer
        self.modelName = modelName
        print(modelName, "; Number of components for PCA: ", n_components, "; Loss function: ", lossfn, transformer)

  

    def trainAndTest(self,trainingPercent=0.80):
        # Creating data frame with X as features and Y as outcome
        def extractFeatures():
            featureset=[]
            for Lastname, Firstname in zip(self.df[self.lastName], self.df[self.firstName]):
                fName = str(Firstname).lower()
                lName = str(Lastname).lower()
                if len(fName) != 1:
                    featureset.append(lName + ' ' + fName)
                else:
                    featureset.append(lName)
        
            return featureset
        
        # get X, Y data
        X = extractFeatures()
        Y = self.df[self.ethnicityCol].values
        data = np.column_stack((X, Y)) 

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, train_size=0.80, random_state=42)
        print('Split training set and testing set 0.80 and 0.20.')

        # Function for customizing how CountVectorizer tokenize strings
        def words_and_char_bigrams(text):
            words = re.findall(r'\w{3,}', text)
            for w in words:
                for i in range(len(w) - 2):
                    yield w[i:i+3]
                for i in range(len(w) - 3):
                    yield w[i:i+4]

        
        # vectorizer for transforming features
        if self.transformer == 'Count':
            vect = CountVectorizer(analyzer=words_and_char_bigrams)
        elif self.transformer == 'Tfid':
            vect = TfidfVectorizer(analyzer=words_and_char_bigrams)
        
        #dimension reduction model 
        svd = TruncatedSVD(n_components=self.n_components)
        normalizer = Normalizer()

        # Pick classfier model
        if self.modelName == "RF":
            clf = RandomForestClassifier(class_weight=self.class_weight, random_state=42)
        elif self.modelName == "SVM":
            clf = SGDClassifier(loss=self.lossfn, penalty=self.penalty,random_state=42)
           
        # Build Pipeline
        pipeline = make_pipeline(vect,svd,normalizer,clf)
        # train model
        model = pipeline.fit(x_train, y_train)
        y_preds = pipeline.predict(x_test)


        ## Results 
        print("** "+self.modelName+" Result **")
        print("Accuracy: ",metrics.accuracy_score(y_test, y_preds))
        print("Confusion Matrix: ")
        cm = metrics.confusion_matrix(y_test, y_preds)
        print(cm)
        sensitivity = float(cm[1][1]) / (cm[1][0]+cm[1][1])
        specificity = float(cm[0][0]) / (cm[0][0]+cm[0][1])
        weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
        print('Sensitivity ',sensitivity, '; Specificity: ', specificity)
        print('Weighted Accuracy: ', weightedAccuracy)
        self.classifaction_report_csv(self.modelName, sensitivity, specificity, float(cm[1][1]), float(cm[1][0]), float(cm[0][0]), float(cm[0][1]) )
        return (self.modelName, pipeline, sensitivity, self.n_components, self.transformer)


    def classifaction_report_csv(self, modelName, sensitivity, specificity, c1correct, c1false, c0correct, c0false):
        if os.path.isfile("record_new.csv"):
            df = pd.read_csv("record_new.csv")
                # TODO: append new data to df
            df2 = pd.DataFrame({'Model':modelName,
                                'Sensitivity': [sensitivity],
                                'Specificity': [specificity],
                                'T1P1': [c1correct],
                                'T1P0': [c1false],
                                'T0P0': [c0correct],
                                'T0P1': [c0false],
                                'n_components': [self.n_components],
                                'Loss function': [self.lossfn],
                                'Class Weight': [self.class_weight],
                                'Penalty': [self.penalty],
                                'Transformer': [self.transformer]})
            df2 = df2[['Model','n_components','Transformer','Loss function','Class Weight','Penalty','Sensitivity','Specificity','T1P1','T1P0','T0P0','T0P1']]
            with open('record_new.csv', 'a') as f:
                df2.to_csv(f, header=False, index=False)
        else:
            #create a new table and save it as file.
            df = pd.DataFrame({'Model':modelName,
                                'Sensitivity': [sensitivity],
                                'Specificity': [specificity],
                                'T1P1': [c1correct],
                                'T1P0': [c1false],
                                'T0P0': [c0correct],
                                'T0P1': [c0false],
                                'n_components': [self.n_components],
                                'Loss function': [self.lossfn],
                                'Class Weight': [self.class_weight],
                                'Penalty': [self.penalty],
                                'Transformer': [self.transformer]})
            df = df[['Model','n_components','Transformer','Loss function','Class Weight','Penalty','Sensitivity','Specificity','T1P1','T1P0','T0P0','T0P1']]
            df.to_csv("record_new.csv", index = False)



## Runing on server 
if platform.system() == 'Linux':
    list_comp = [200, 300, 500, 800, 1000, 1500, 2000, 3000]
    transformers = ['Count', 'Tfid']
    models = ['RF', 'SVM']
    best_sensitivity = 0.0
    best_ppl = None
    model = None
    n_comp = None
    tf = None
    for n_components in list_comp:
        for transformer in transformers:
            for model in models:
                df = ethPredictor("1910p_10percent.csv", "LName","Japanese", n_components, "balanced_subsample", "hinge", "l2",transformer, model)
                result = df.trainAndTest()
                if result[2]>best_sensitivity:
                    best_sensitivity = result[2]
                    best_ppl = result[1]
                    model = result[0]
                    n_comp= result[3]
                    tf = result[4]
    with open('clf.pkl', 'wb') as out_strm: 
        dill.dump(best_ppl, out_strm)
    print("Saved best classifier as as clf.pkl with sensitivity score: ",model,best_sensitivity, n_comp, tf)


else:
    list_comp = [200, 300, 500]
    transformers = ['Count', 'Tfid']
    models = ['RF', 'SVM']
    best_sensitivity = 0.0
    best_ppl = None
    modelName = None
    n_comp = None
    tf = None
    for n_components in list_comp:
        for transformer in transformers:
            for model in models:
                df = ethPredictor("5percentdata.csv", "LName","Japanese", n_components, "balanced_subsample", "hinge", "l2", transformer, model)
                result = df.trainAndTest()
                if result[2]>best_sensitivity:
                    best_sensitivity = result[2]
                    best_ppl = result[1]
                    modelName = result[0]
                    n_comp= result[3]
                    tf = result[4]
    with open('clf.pkl', 'wb') as out_strm: 
        dill.dump(best_ppl, out_strm)
    print("Saved best classifier as clf.pkl with sensitivity score: ",model,best_sensitivity, n_comp, tf)





