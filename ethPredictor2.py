
import numpy as np
import pandas as pd
import random
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier



class ethPredictor2():

    ## colName: the name of the column for names (Lastname, Firstname)

    ## ethCol: the name of the column that contains ethnicity information.

    ## ethnicityCol: the name of dummy variable column that indicates the interested etnnicity.

    ## Read in data with specified variables.

    def __init__(self, fileName, lastName, ethnicityCol,n_components, class_weight, lossfn, penalty, firstName='FName'):
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
        print("Tuning params: number of components for PCA: ", n_components, "; Loss function: ", lossfn)

  


    def trainAndTest(self,trainingPercent=0.80):
        featureset=[]
        for Lastname, Firstname in zip(self.df[self.lastName], self.df[self.firstName]):
            featureset.append(str(Lastname).lower() + ' ' + str(Firstname).lower())
        
        Y = self.df[self.ethnicityCol].values

        # Function for customizing how CountVectorizer tokenize strings
        def words_and_char_bigrams(text):
            words = re.findall(r'\w{3,}', text)
            for w in words:
                for i in range(len(w) - 2):
                    yield w[i:i+3]
                for i in range(len(w) - 3):
                    yield w[i:i+4]

        data = np.column_stack((featureset, Y))    

        vect = CountVectorizer(analyzer=words_and_char_bigrams)
        
        X = vect.fit_transform(featureset).toarray()
        print("* shape of the CountVectorizer: ", X.shape)



        # dimension reduction here 
        # principle component analysis

        svd = TruncatedSVD(n_components=self.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X) 
        print("* shape of the pca components: ", X.shape)
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

        
        names = vect.get_feature_names()
        print('Created Features: ', names[:7])
        #print('Numerical features: ', X[:7])

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, train_size=0.80, random_state=42)
        print('Split training set and testing set 0.80 and 0.20.')
   
        ### Create string vars for balanced_subsamp,e and hinge and penalty on top ###
        clf_RF = RandomForestClassifier(class_weight=self.class_weight, random_state=42)
        clf_RF.fit(x_train, y_train)
        
        clf_SVM = SGDClassifier(loss=self.lossfn, penalty=self.penalty,random_state=42)
        clf_SVM.fit(x_train, y_train)

        RF_preds = clf_RF.predict(x_test)
        SVM_preds = clf_SVM.predict(x_test)
       

        ## Save report to a csv file
        def classifaction_report_csv(modelName, sensitivity, specificity, c1correct, c1false, c0correct, c0false):
            if os.path.isfile("record.csv"):
                df = pd.read_csv("record.csv")
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
                                    'Penalty': [self.penalty]})
                df2 = df2[['Model','n_components','Loss function','Class Weight','Penalty','Sensitivity','Specificity','T1P1','T1P0','T0P0','T0P1']]
                df=df.append(df2, ignore_index=True)
                with open('record.csv', 'w') as f:
                    df.to_csv(f, header=True)
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
                                    'Penalty': [self.penalty]})
                df = df[['Model','n_components','Loss function','Class Weight','Penalty','Sensitivity','Specificity','T1P1','T1P0','T0P0','T0P1']]
                df.to_csv("record.csv", index = False)  
        

        ## Results of Random forest and SVM
        print("** Random Forest Result **")
        print("Accuracy: ",metrics.accuracy_score(y_test, RF_preds))
        print("Confusion Matrix: ")
        cm = metrics.confusion_matrix(y_test, RF_preds)
        print(cm)
        sensitivity = float(cm[1][1]) / (cm[1][0]+cm[1][1])
        specificity = float(cm[0][0]) / (cm[0][0]+cm[0][1])
        weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
        print('Sensitivity ',sensitivity, '; Specificity: ', specificity)
        print('Weighted Accuracy: ', weightedAccuracy)
        classifaction_report_csv("RF", sensitivity, specificity, float(cm[1][1]), float(cm[1][0]), float(cm[0][0]), float(cm[0][1]) )

        topfeatures= []
        feature_names = vect.get_feature_names()
        top10 = np.argsort(clf_RF.feature_importances_)[-10:]
        for i in top10:
            topfeatures.append(feature_names[i]) 
        print("Random Forest Most informative features: ")
        print(topfeatures)

        print('***********************************')

        print("** SVM Result **")
        print("Accuracy: ",metrics.accuracy_score(y_test, SVM_preds))
        print("Confusion Matrix: ")
        cm2 = metrics.confusion_matrix(y_test,SVM_preds)
        print(cm2)
        sensitivity = float(cm2[1][1]) / (cm2[1][0]+cm2[1][1])
        specificity = float(cm2[0][0]) / (cm2[0][0]+cm2[0][1])
        weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
        print('Sensitivity ',sensitivity, '; Specificity: ', specificity)
        print('Weighted Accuracy: ', weightedAccuracy)
        classifaction_report_csv("SVM", sensitivity, specificity, float(cm2[1][1]), float(cm2[1][0]), float(cm2[0][0]), float(cm2[0][1]))
  
        ## Get most informative featrues: 
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


list_comp = [200, 300, 500, 800, 1000, 1500, 2000, 3000]
for n_components in list_comp:
    df = ethPredictor2("5percentdata.csv", "LName","Japanese", n_components, "balanced_subsample", "hinge", "l2")
    df.trainAndTest()
