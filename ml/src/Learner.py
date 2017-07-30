import os
import re
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

import simplejson
from utils import Utilities
from itertools import takewhile, izip

class Learner:
    global logger
    class LabelledDocs:
        def __init__(self, doc, label):
            self.doc = doc
            self.label = label

    @staticmethod
    def dir2jsons(json_dir):
        jsons = []
        for root, dirs, files in os.walk(json_dir, topdown=False):
            for filename in files:
                if re.search('json$', filename):
                    with open(os.path.join(root, filename), "rb") as fin:
                        try:
                            jsons.append(simplejson.load(fin))
                        except Exception as e:
                            Utilities.logger.error(e)
        return jsons

    @staticmethod
    def same_prefix(str_a, str_b):
        for i, c in enumerate(str_a):
            if i > 6:
                return True
            if c == str_b[i]:
                continue
            else:
                return False


    @staticmethod
    def feature_filter_by_prefix(vocab, docs):
        examined = []
        for i in range(len(vocab)):
            logger.info('i: ' + vocab[i] + ' ' +str(i))
            if len(vocab[i]) < 6 or vocab[i] in examined:
                continue
            for j in range(i + 1, len(vocab)):
                #logger.info('j: ' + vocab[j] + ' ' + str(j))
                if (len(vocab[j]) < 6):
                    examined.append(vocab[j])
                    continue
                if vocab[i] in vocab[j] or vocab[j] in vocab[i]: #Learner.same_prefix(vocab[i], vocab[j]):
                    #logger.info('Found ' + vocab[i] + ' ' + vocab[j] + ' ' + str(i))
                    examined.append(vocab[j])
                    for doc in docs:
                        if vocab[j] in doc.doc:
                            doc.doc = str(doc.doc).replace(vocab[j], vocab[i])
        instances = []
        labels = []
        for doc in docs:
            instances.append(doc.doc)
            labels.append(doc.label)
        vectorizer = CountVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=None, \
                                     max_features=100000)
        train_data = vectorizer.fit_transform(instances)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        # train_data = train_data.toarray()
        logger.info(train_data.shape)
        return train_data, labels

    @staticmethod
    def gen_instances(pos_json_dir, neg_json_dir):
        pos_jsons = Learner.dir2jsons(pos_json_dir)
        neg_jsons = Learner.dir2jsons(neg_json_dir)
        logger.info(len(pos_jsons))
        logger.info(len(neg_jsons))
        docs = Learner.gen_docs(pos_jsons)
        docs = docs + (Learner.gen_docs(neg_jsons))
        instances = []
        labels = []
        for doc in docs:
            instances.append(doc.doc)
            labels.append(doc.label)
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=None, \
                                     max_features=100000)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        train_data = vectorizer.fit_transform(instances)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        # train_data = train_data.toarray()
        logger.info(train_data.shape)
        # Take a look at the words in the vocabulary
        vocab = vectorizer.get_feature_names()
        # logger.info(vocab)
        #train_data, labels = Learner.feature_filter_by_prefix(vocab, docs)

        return train_data, labels

    @staticmethod
    def train(train_data, labels):
        # Initialize a Random Forest classifier with 100 trees
        cv = KFold(n_splits=10, random_state=33, shuffle=True)
        clf = DecisionTreeClassifier(class_weight='balanced')
        logger.info(cross_val_score(clf, train_data, labels, cv=cv))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        #forest = clf.fit(train_data, labels)


    @staticmethod
    def gen_docs(jsons):
        docs = []
        for flow in jsons:
            label = flow['label']
            line = ''
            line += flow['domain']
            line += flow['uri']
            docs.append(Learner.LabelledDocs(line, label))
        return docs


if __name__ == '__main__':
    logger = Utilities.set_logger('Learner')
    data, labels = Learner.gen_instances("C:\\Users\\hfu\\Documents\\flows\\CTU-13\\CTU-13-1\\1",
                "C:\\Users\\hfu\\Documents\\flows\\CTU-13\\CTU-13-1\\0")
        #"C:\Users\hfu\IdeaProjects\\recon\\test\\1",
         #                 "C:\Users\hfu\IdeaProjects\\recon\\test\\0")
    Learner.train(data, labels)
