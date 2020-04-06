import numpy as np
import pandas as pd

#For other warnings
import warnings
warnings.filterwarnings("ignore")

#https://stackoverflow.com/questions/879173/how-to-ignore-deprecation-warnings-in-python
#For warnings related to joblib
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import joblib

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

###################################################
######## Loading the generated csv file ###########

For_deployment = pd.read_csv('C:\\Users\\Bhavesh\\Desktop\\BE Project\\Project Files\\For_deployment.csv')

######## Creating the tfidf_vec pkl file ###########
### And Dumping the vectorizer in tfidf_vect.pkl (Medical Text) ####

tfidf_vect = TfidfVectorizer(min_df=10)
tfidf_vect.fit(For_deployment['Text'].values)

joblib.dump(tfidf_vect, 'tfidf_vect.pkl')
X_train_text = tfidf_vect.transform(For_deployment['Text'].values)

######## Creating the bow_vec pkl file ###########
### And Dumping the vectorizer in bow_vec.pkl (Medical Text) ####

bow_vect = CountVectorizer(min_df=10)
bow_vect.fit(For_deployment['Text'].values)

joblib.dump(bow_vect, 'bow_vect.pkl')
X_train_text_bow = bow_vect.transform(For_deployment['Text'].values)

######## Creating the count_vec_gene pkl file ###########
### And Dumping the OHE in count_vect_gene.pkl (Medical Text) ####

count_vect_gene = CountVectorizer(binary=True)
count_vect_gene.fit(For_deployment['Gene'].values)

joblib.dump(count_vect_gene, 'count_vect_gene.pkl')
X_train_gene = count_vect_gene.transform(For_deployment['Gene'].values)

######## Creating the count_vec_var pkl file ###########
### And Dumping the OHE in count_vect_var.pkl (Medical Text) ####

count_vect_var = CountVectorizer(binary=True)
count_vect_var.fit(For_deployment['Variation'].values)

joblib.dump(count_vect_var, 'count_vect_var.pkl')
X_train_var = count_vect_var.transform(For_deployment['Variation'].values)
Y = For_deployment['Class'].values

# Concatenating for TFIDF
X_train_tfidf = hstack((X_train_text, X_train_gene, X_train_var)).tocsr()

# Concatenating for BoW
X_train_bow = hstack((X_train_text_bow, X_train_gene, X_train_var)).tocsr()

##### TFIDF LOGISTIC REGRESSION ####

clf_tfidf = LogisticRegression(C=10, class_weight='balanced')
clf_tfidf.fit(X_train_tfidf, Y)

clf = CalibratedClassifierCV(clf_tfidf)
clf.fit(X_train_tfidf, Y)
joblib.dump(clf, 'LogReg_tfidf.pkl')

##### TFIDF Naive Bayes ####

clf_tfidf_nb = MultinomialNB(alpha = 0.00001)
clf_tfidf_nb.fit(X_train_tfidf, Y)

clf_nb = CalibratedClassifierCV(clf_tfidf_nb)
clf_nb.fit(X_train_tfidf, Y)
joblib.dump(clf_nb, 'NaiveBayes_tfidf.pkl')

##### TFIDF knn ####

clf_tfidf_knn = KNeighborsClassifier(n_neighbors = 13, n_jobs = -1)
clf_tfidf_knn.fit(X_train_tfidf, Y)

clf_knn = CalibratedClassifierCV(clf_tfidf_knn)
clf_knn.fit(X_train_tfidf, Y)
joblib.dump(clf_knn, 'KNN_tfidf.pkl')

##### TFIDF RF ####

clf_tfidf_rf = RandomForestClassifier(n_estimators=500, max_depth=7, class_weight='balanced')
clf_tfidf_rf.fit(X_train_tfidf, Y)

clf_rf = CalibratedClassifierCV(clf_tfidf_rf)
clf_rf.fit(X_train_tfidf, Y)
joblib.dump(clf_rf, 'RandomForest_tfidf.pkl')

############################################################################

##### BOW LOGISTIC REGRESSION ####

bow_clf_lr = LogisticRegression(C=10, class_weight='balanced')
bow_clf_lr.fit(X_train_bow, Y)

bow_lr = CalibratedClassifierCV(bow_clf_lr)
bow_lr.fit(X_train_bow, Y)
joblib.dump(bow_lr, 'LogReg_bow.pkl')

##### BOW Naive Bayes ####

clf_bow_nb = MultinomialNB(alpha = 0.00001)
clf_bow_nb.fit(X_train_bow, Y)

bow_nb = CalibratedClassifierCV(clf_bow_nb)
bow_nb.fit(X_train_bow, Y)
joblib.dump(bow_nb, 'NaiveBayes_bow.pkl')

##### BOW knn ####

clf_bow_knn = KNeighborsClassifier(n_neighbors = 13, n_jobs = -1)
clf_bow_knn.fit(X_train_bow, Y)

bow_knn = CalibratedClassifierCV(clf_bow_knn)
bow_knn.fit(X_train_bow, Y)
joblib.dump(bow_knn, 'KNN_bow.pkl')

##### BOW RF ####

clf_bow_rf = RandomForestClassifier(n_estimators=500, max_depth=7, class_weight='balanced')
clf_bow_rf.fit(X_train_bow, Y)

bow_rf = CalibratedClassifierCV(clf_bow_rf)
bow_rf.fit(X_train_bow, Y)
joblib.dump(bow_rf, 'RandomForest_bow.pkl')