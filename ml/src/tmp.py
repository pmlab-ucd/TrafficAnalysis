from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import PorterStemmer
import nltk

'''
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        #return lambda doc: word_ngrams([stemmer.stem(w) for w in analyzer(doc)],
          #                             ngram_range=self.ngram_range, stop_words=self.stop_words)
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(6, 8))
counts = vectorizer.fit_transform(['appverify', 'clientnew', 'www'])
print len(vectorizer.get_feature_names()), vectorizer.get_feature_names()
print vectorizer.transform(['appvetd'])
print len(vectorizer.get_feature_names()), vectorizer.get_feature_names()


ch2 = SelectKBest(chi2, k=3)
X_new = ch2.fit_transform(counts, [1, 1, 0])
feature_names = vectorizer.get_feature_names()
if feature_names != None:
        feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]

dict = np.asarray(vectorizer.get_feature_names())[ch2.get_support()]
print feature_names
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(6, 8), vocabulary=feature_names)
#vectorizer._validate_vocabulary()
print vectorizer.transform(['appverify'])
print len(vectorizer.get_feature_names()), vectorizer.get_feature_names()




vectorizer = StemmedCountVectorizer(analyzer='char_wb', ngram_range=(6, 8))
counts = vectorizer.fit_transform(['appverify', 'appverifying', 'clientnew', 'www'])
print len(vectorizer.get_feature_names()), vectorizer.get_feature_names()
print vectorizer.transform(['appverify'])
'''
vectorizer = TfidfVectorizer(analyzer='word')
vectorizer.fit_transform(['www.appverify'])
tokens = vectorizer.get_feature_names()
print vectorizer.transform(['haha', 'www'])
print tokens