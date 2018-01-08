from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.svm import SVC
import Utils

def get_pipeline():
    vectorizer = CountVectorizer(tokenizer = Utils.spacy_tokenizer, ngram_range=(1,1)) 
    classifier = SVC(C=10, kernel='linear')

    pipe = Pipeline([("cleaner", Utils.predictors()),
                     ('vectorizer', vectorizer),
                     ('tfidftransformer', TfidfTransformer()),
                     ('classifier', classifier)])
    return pipe


