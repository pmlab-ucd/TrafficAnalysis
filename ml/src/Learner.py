import os
import re
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn import tree
import pydotplus
import codecs
import cPickle
from sklearn.metrics import accuracy_score, precision_score
from sklearn import svm

import simplejson
from utils import Utilities
from itertools import takewhile, izip

import string
import random

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

class Learner:
    global logger
    class LabelledDocs:
        def __init__(self, doc, label):
            self.doc = doc
            self.label = label

    @staticmethod
    def dir2jsons(json_dir):
        jsons = []
        if json_dir == None:
            return jsons
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
    def gen_instances(pos_json_dir, neg_json_dir, output_dir=os.curdir, to_vec=True, simulate=False):
        pos_jsons = Learner.dir2jsons(pos_json_dir)
        neg_jsons = Learner.dir2jsons(neg_json_dir)
        logger.info('lenPos: ' + str(len(pos_jsons)))
        logger.info('lenNeg: ' + str(len(neg_jsons)))
        docs = Learner.gen_docs(pos_jsons)
        docs = docs + (Learner.gen_docs(neg_jsons))
        if simulate:
            if len(neg_jsons) == 0:
                docs = docs + Learner.simulate_flows(len(pos_jsons), 0)
        instances = []
        labels = []
        for doc in docs:
            instances.append(doc.doc)
            labels.append(doc.label)
        if not to_vec:
            return  instances, labels
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
        # Save vectorizer.vocabulary_
        cPickle.dump(vectorizer.vocabulary_, open(output_dir + '/' + "vocabulary.pkl", "wb"))

        return train_data, labels, vocab

    @staticmethod
    def ocsvm(train_data, labels, output_dir=os.curdir):
        nu = float(labels.count(0)) / len(labels)

        clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)
        clf.fit(train_data)

        with open(output_dir + '/' + 'ocsvm.pkl', 'wb') as fid:
            cPickle.dump(clf, fid)

    @staticmethod
    def train_tree(train_data, labels, feature_names=None, output_dir=os.curdir, tree_name='tree'):
        # Initialize a Random Forest classifier with 100 trees
        cv = KFold(n_splits=10, random_state=33, shuffle=True)
        clf = DecisionTreeClassifier(class_weight='balanced')
        results = cross_val_score(clf, train_data, labels, cv=cv, scoring='f1')
        logger.info(results)

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)
        dot_data = tree.export_graphviz(clf, out_file=output_dir + '/' + tree_name +'.dot', feature_names=feature_names,
                                        label='root', impurity=False, special_characters=True, max_depth=5)
        dotfile = open(output_dir + '/' + tree_name +'.dot', 'r')
        graph = pydotplus.graph_from_dot_data(dotfile.read())
        graph.write_pdf(output_dir + '/' + tree_name +'.pdf')
        dotfile.close()
        simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
                        separators=(',', ':'), sort_keys=True, indent=4)
        # save the classifier
        with open(output_dir + '/' + 'classifier.pkl', 'wb') as fid:
            cPickle.dump(clf, fid)

    @staticmethod
    def rand_str(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    @staticmethod
    def simulate_flows(size, label):
        docs = []
        for _ in range(size):
            docs.append(Learner.LabelledDocs(Learner.rand_str(), label))
        return docs

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

    @staticmethod
    def predict(model, voc, instances, labels=None):
        loaded_vec = CountVectorizer(decode_error="replace", vocabulary=voc)
        data = loaded_vec.fit_transform(instances)
        y_1 = model.predict(data)
        logger.info(y_1)
        if labels:
            logger.info(accuracy_score(labels, y_1))

if __name__ == '__main__':
    logger = Utilities.set_logger('Learner')
    base_dir = 'C:\\Users\\hfu\\Documents\\flows\\CTU-13\\'
    dataset_num = '2'
    dataset = 'CTU-13-' + dataset_num + '\\'


    train = True
    learner = 'tree'
    if train:
        #"C:\Users\hfu\IdeaProjects\\recon\\test\\1",
         #                 "C:\Users\hfu\IdeaProjects\\recon\\test\\0")
        if learner == 'tree':
            classifier_dir = base_dir + dataset
            vocab_dir = base_dir + dataset
            simulate = True
            if simulate:
                classifier_dir = base_dir + 'CTU-13-1\\' + '1'
                vocab_dir = base_dir + 'CTU-13-1\\' + '1'
                data, labels, feature_names = Learner.gen_instances(base_dir + 'CTU-13-1\\' + '1',
                                                                    None,
                                                                    output_dir=vocab_dir, simulate=simulate)
                Learner.train_tree(data, labels, feature_names=feature_names, output_dir=classifier_dir,
                                   tree_name='Fig_tree_normal')
            else:
                data, labels, feature_names = Learner.gen_instances(base_dir + 'CTU-13-1\\' + '\\1',
                                                                base_dir + dataset + '\\0',
                                                                output_dir=vocab_dir, simulate=False)
                Learner.train_tree(data, labels, feature_names=feature_names, output_dir=classifier_dir,
                                   tree_name='Fig_tree_' + dataset_num)
        elif learner == 'ocsvm':
            classifier_dir = base_dir + 'CTU-13-1\\' + '\\1'
            vocab_dir = base_dir + 'CTU-13-1\\' + '\\1'
            data, labels, feature_names = Learner.gen_instances(vocab_dir,
                                                                base_dir + 'CTU-13-1\\' + '\\0\\CC',
                                                                output_dir=vocab_dir)
            Learner.ocsvm(data, labels, output_dir=classifier_dir)
    else:
        if learner == 'tree':
            simulate = True
            classifier_dir = base_dir + dataset
            vocab_dir = base_dir + dataset
            if simulate:
                classifier_dir = base_dir + 'CTU-13-1\\' + '\\1'
                vocab_dir = base_dir + 'CTU-13-1\\' + '\\1'
            learner_path = classifier_dir + '\\' + 'classifier.pkl'
        elif learner == 'ocsvm':
            classifier_dir = base_dir + 'CTU-13-1\\' + '\\1'
            vocab_dir = base_dir + 'CTU-13-1\\' + '\\1'
            learner_path = classifier_dir + '\\' + 'ocsvm.pkl'

        data, labels = Learner.gen_instances('', base_dir + dataset + '\\0', to_vec=False)
        if learner == 'ocsvm':
            labels = [-1.] * len(labels)
        Learner.predict(cPickle.load(open(learner_path, 'rb')),
                            cPickle.load(open(vocab_dir + '\\' + 'vocabulary.pkl', "rb")), data, labels=labels)

