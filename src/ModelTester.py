from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
import numpy as np
import time, csv

import string
punctuations = string.punctuation

import spacy
parser = spacy.load("en")

#Replace this with your training data location
training_data = r'C:\projects\manualExtraction\Model\trainingdata.tsv'

#Custom transformer using spaCy 
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# Basic utility function to clean the text 
def clean_text(text):     
    return text.strip().lower()

#Create spacy tokenizer that parses a sentence and generates tokens
#these can also be replaced by word vectors 
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens

class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

class MeanEmbeddingVectorizer(object):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([parser(w).vector for w in words], axis=0)
            for words in X
        ])

#create vectorizer object to generate feature vectors, we will use custom spacy’s tokenizer
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1)) 

# Load sample data
all_data=[]
try:
    with open(training_data, 'r') as tsvin:
                tsvin = csv.reader(tsvin, delimiter='\t')
                for row in tsvin:
                    all_data.append((row[0], row[1]))
except IOError:
        print("Could not read file:" + training_data)

#grid search

grid = [{'classifier__C': [1, 10, 100, 1000],'classifier__kernel': ['linear']}, {'classifier__kernel': ['rbf'], 'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001]}]

pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('tfidftransformer', TfidfTransformer()),
                 ('classifier', SVC())])

print(pipe.get_params().keys())
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
clf = GridSearchCV(pipe, grid, cv=cv)
start = time.clock()
clf.fit([x[0] for x in all_data], [x[1] for x in all_data])
print("time: " + str(time.clock()-start))
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()

names = ["Nearest Neighbors", "Linear SVM", "SVC10", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    LinearSVC(),
    SVC(C=10, kernel='linear'),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


for i in range(len(classifiers)):
    start = time.clock()
    pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('to_dense', DenseTransformer()),
                 ('classifier', classifiers[i])])
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(pipe, [x[0] for x in all_data], [x[1] for x in all_data], cv=cv)
    print(scores)
    print("Classifier:%s Accuracy: %0.2f (+/- %0.2f) %f" % (names[i], scores.mean(), scores.std() * 2, time.clock() - start))


print("With tfidfvectorizer")

for i in range(len(classifiers)):
    start = time.clock()
    pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('tfidftransformer', TfidfTransformer()),
                 ('to_dense', DenseTransformer()),
                 ('classifier', classifiers[i])])
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(pipe, [x[0] for x in all_data], [x[1] for x in all_data], cv=cv)
    print(scores)
    print("Classifier:%s Accuracy: %0.2f (+/- %0.2f) %f" % (names[i], scores.mean(), scores.std() * 2, time.clock() - start))

print("With meanEmbeddingVectorizer")

for i in range(len(classifiers)):
    start = time.clock()
    pipe = Pipeline([("cleaner", predictors()),
                 ('embedding', MeanEmbeddingVectorizer()),
                 ('classifier', classifiers[i])])
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(pipe, [x[0] for x in all_data], [x[1] for x in all_data], cv=cv)
    print(scores)
    print("Classifier:%s Accuracy: %0.2f (+/- %0.2f) %f" % (names[i], scores.mean(), scores.std() * 2, time.clock() - start))

#Try guassian process and linear with different parameters

print("done!!")