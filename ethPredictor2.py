
import numpy as np
import pandas as pd
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

class ethPredictor2():

    ## colName: the name of the column for names (Lastname, Firstname)

    ## ethCol: the name of the column that contains ethnicity information.

    ## ethnicityCol: the name of dummy variable column that indicates the interested etnnicity.

    ## Read in data with specified variables.
    def __init__(self, fileName, lastName, ethnicityCol,firstName='FName'):
        print ("Reading data from file: " + fileName)
        self.df = pd.read_csv(fileName, encoding='latin-1')
        self.lastName = lastName
        self.firstName = firstName
        self.ethnicityCol = ethnicityCol
        self.ratio = self.df.groupby(ethnicityCol).count()
        print(self.ratio)


    def trainAndTest(self,trainingPercent=0.80):
        featureset=[]
        for Lastname, Firstname in zip(self.df[self.lastName], self.df[self.firstName]):
            featureset.append(str(Lastname).lower() + ' ' + str(Firstname).lower())
        
        # Function for customizing how CountVectorizer tokenize strings
        def words_and_char_bigrams(text):
            words = re.findall(r'\w{3,}', text)
            for w in words:
                for i in range(len(w) - 2):
                    yield w[i:i+3]
                for i in range(len(w) - 3):
                    yield w[i:i+4]

        vect = CountVectorizer(analyzer=words_and_char_bigrams)
        X = vect.fit_transform(featureset).toarray()

        #vect = HashingVectorizer(analyzer=words_and_char_bigrams)
       
        # dimension reduction here 
        # principle component analysis
        
        Y = self.df[self.ethnicityCol].values
        
        names = vect.get_feature_names()
        print('Created Features: ', names[:7])
        print('Numerical features: ', X[:7])

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, train_size=0.80)
        print('Split training set and testing set 0.80 and 0.20.')
   
        clf_RF = RandomForestClassifier(class_weight="balanced_subsample")
        clf_RF.fit(x_train, y_train)
        
        clf_SVM = SGDClassifier(loss='hinge', penalty='l2',random_state=42)
        clf_SVM.fit(x_train, y_train)

        RF_preds = clf_RF.predict(x_test)
        SVM_preds = clf_SVM.predict(x_test)
       
        print("Random Forest Result:")
        print("Accuracy: ",metrics.accuracy_score(y_test, RF_preds))
        print("Confusion Matrix: ")
        cm = metrics.confusion_matrix(y_test, RF_preds)
        print(cm)
        sensitivity = float(cm[1][1]) / (cm[1][0]+cm[1][1])
        specificity = float(cm[0][0]) / (cm[0][0]+cm[0][1])
        weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
        print('Sensitivity ',sensitivity, '; Specificity: ', specificity)
        print('Weighted Accuracy: ', weightedAccuracy)

        topfeatures= []
        feature_names = vect.get_feature_names()
        top10 = np.argsort(clf_RF.feature_importances_)[-10:]
        for i in top10:
            topfeatures.append(feature_names[i]) 
        print("Random Forest Most informative features: ")
        print(topfeatures)

        print('***********************************')

        print("SVM Result:")
        print("Accuracy: ",metrics.accuracy_score(y_test, SVM_preds))
        print("Confusion Matrix: ")
        cm2 = metrics.confusion_matrix(y_test,SVM_preds)
        print(cm2)
        sensitivity = float(cm2[1][1]) / (cm2[1][0]+cm2[1][1])
        specificity = float(cm2[0][0]) / (cm2[0][0]+cm2[0][1])
        weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
        print('Sensitivity ',sensitivity, '; Specificity: ', specificity)
        print('Weighted Accuracy: ', weightedAccuracy)

  
        coefs_with_fns = sorted(zip(clf_SVM.coef_[0], feature_names))
        top = zip(coefs_with_fns[:10], coefs_with_fns[:-11:-1])
        print("Most informative features: ")
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

        # Save trained classfier
        save_classifier = open("randomForest.pickle","wb")
        pickle.dump(clf_RF, save_classifier)
        save_classifier.close()
        print("Saved Random Forest classifier as randomForest.pickle")



df = ethPredictor2("5percentdata.csv", "LName","Japanese")
df.trainAndTest()
