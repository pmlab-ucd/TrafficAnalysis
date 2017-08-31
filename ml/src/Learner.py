import os
import re
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
import pydotplus
import codecs
import cPickle
from sklearn.metrics import accuracy_score, precision_score
from sklearn import svm
import numpy as np
from sklearn.linear_model import LogisticRegression
from time import time

import simplejson
import json
from utils import Utilities
from itertools import takewhile, izip

import string
import random

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline

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
                if '201' in filename and re.search('json$', filename):
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
    def gen_instances(pos_json_dir, neg_json_dir, to_vec=True, simulate=False):
        pos_jsons = Learner.dir2jsons(pos_json_dir)
        neg_jsons = Learner.dir2jsons(neg_json_dir)
        logger.info('lenPos: ' + str(len(pos_jsons)))
        logger.info('lenNeg: ' + str(len(neg_jsons)))
        docs = Learner.gen_docs(pos_jsons, 1)
        docs = docs + (Learner.gen_docs(neg_jsons, -1))
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

        return train_data, labels, vocab, vectorizer

    @staticmethod
    def ocsvm(train_data, labels, output_dir=os.curdir):
        nu = float(labels.count(0)) / len(labels)

        clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)
        clf.fit(train_data)

        with open(output_dir + '/' + 'ocsvm.pkl', 'wb') as fid:
            cPickle.dump(clf, fid)

    @staticmethod
    def train_bayes(train_data, labels, cross_vali=True, feature_names=None):
        clf = BernoulliNB()
        results = None
        if cross_vali == True:
            results = Learner.cross_validation(clf, train_data, labels)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Bayes: ' + str(results))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        return clf, results

    @staticmethod
    def cross_validation(clf, data, labels, scoring='f1', n_splits=5):
        t0 = time()
        results = dict()
        cv = KFold(n_splits=5, shuffle=True)

        cv_res = cross_val_score(clf, data, labels, cv=cv, scoring='f1').tolist()
        # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
        # separators=(',', ':'), sort_keys=True, indent=4)
        duration = time() - t0
        results['duration'] = duration
        results['cv_res'] = cv_res
        results['cv_res_mean'] = sum(cv_res) / n_splits
        return results


    @staticmethod
    def train_SVM(train_data, labels, cross_vali=True, feature_names=None):
        clf = svm.SVC(class_weight='balanced')
        results = None
        if cross_vali == True:
            results = Learner.cross_validation(clf, train_data, labels)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('SVM: ' + str(results))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        return clf, results

    @staticmethod
    def train_logistic(train_data, labels, cross_vali=True, feature_names=None):
        clf = LogisticRegression(class_weight='balanced')
        results = None
        if cross_vali == True:
            results = Learner.cross_validation(clf, train_data, labels)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Logistic: ' + str(results))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        return clf, results

    @staticmethod
    def train_tree(train_data, labels, cross_vali=True, feature_names=None, output_dir=os.curdir, tree_name='tree'):
        clf = DecisionTreeClassifier(class_weight='balanced')
        results = None
        if cross_vali == True:
            results = Learner.cross_validation(clf, train_data, labels)
            #simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
                            #separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Tree: ' + str(results))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        dot_data = tree.export_graphviz(clf, out_file=output_dir + '/' + tree_name +'.dot', feature_names=feature_names,
                                        label='root', impurity=False, special_characters=True) #, max_depth=5)
        dotfile = open(output_dir + '/' + tree_name +'.dot', 'r')
        graph = pydotplus.graph_from_dot_data(dotfile.read())
        graph.write_pdf(output_dir + '/' + tree_name +'.pdf')
        dotfile.close()

        return clf, results

    @staticmethod
    def rand_str(size=6, chars=string.ascii_uppercase + string.digits):
        url = ''.join(random.choice(chars) for _ in range(size))
        if url[0] < 'k':
            url = url + 'net'
        else:
            url = url + 'com'
        url = 'www.' + url
        return url

    @staticmethod
    def simulate_flows(size, label):
        docs = []
        for _ in range(size):
            docs.append(Learner.LabelledDocs('www.' + Learner.rand_str() + '', label))
        return docs

    @staticmethod
    def tree_info(clf):
        info = dict()
        n_nodes = clf.tree_.node_count
        # children_left = clf.tree_.children_left
        # children_right = clf.tree_.children_right
        # feature = clf.tree_.max_features
        # n_feature = clf.tree_.n_features_
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        depth = clf.tree_.max_depth
        info['n_nodes'] = n_nodes
        info['depth'] = depth
        logger.info(info)
        return info


    @staticmethod
    def gen_docs(jsons, label):
        docs = []
        for flow in jsons:
            label = label #flow['label']
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

    @staticmethod
    def feature_selection(X, y, k, count_vectorizer, feature_names=None):
        ch2 = SelectKBest(chi2, k=k)
        X_new = ch2.fit_transform(X, y)
        if feature_names != None:
            feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
        dict = np.asarray(count_vectorizer.get_feature_names())[ch2.get_support()]
        count_vectorizer = CountVectorizer(analyzer="word", vocabulary=dict)
        # cPickle.dump(count_vectorizer.vocabulary, open(output_dir + '/' + "vocabulary.pkl", "wb"))
        return X_new, feature_names, count_vectorizer

    @staticmethod
    def pipe_feature_selection(X, y):
        clf = Pipeline([
            ('feature_selection', SelectKBest(chi2, k=2).fit_transform(X, y)),
            ('classification', RandomForestClassifier())
        ])
        clf.fit(X, y)

    @staticmethod
    def cmp_feature_selection(data_path, output_dir, dataset=None):
        data, labels, feature_names, vec = Learner.gen_instances('C:\Users\hfu\Documents\\flows\\normal\\March',
                                                                 data_path, simulate=False)
        back = [data, labels, feature_names, vec]

        Learner.save2file(vec.vocabulary_, output_dir + '/' + "vocabulary.pkl")
        logger.info(data.shape)
        clf, cv = Learner.train_tree(data, labels, cross_vali=True, feature_names=feature_names,
                                     tree_name='Fig_tree_' + dataset, output_dir=output_dir)
        Learner.save2file(clf, classifier_dir + '\\' + 'classifier.pkl')

        clf_info = Learner.tree_info(clf)
        clf_info['cv'] = cv

        simplejson.dump(clf_info, codecs.open(output_dir + '/tree_info.json', 'w', encoding='utf-8'))

        data, labels, feature_names, vec = back
        data, feature_names, vec = Learner.feature_selection(data, labels, 200,  vec,
                                                             feature_names=feature_names)
        Learner.save2file(vec.vocabulary, output_dir + '/' + "vocabulary_sel.pkl")
        logger.info(data.shape)
        clf, cv = Learner.train_tree(data, labels, cross_vali=True, feature_names=feature_names,
                                     tree_name='Fig_tree_sel_' + dataset, output_dir=output_dir)
        Learner.save2file(clf, classifier_dir + '\\' + 'classifier_sel.pkl')

        clf_info = Learner.tree_info(clf)
        clf_info['cv'] = cv

        json.dump(clf_info, codecs.open(output_dir + '/tree_info_sel.json', 'w', encoding='utf-8'))


        #simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'))
        # separators=(',', ':'), sort_keys=True, indent=4)

    @staticmethod
    def save2file(obj, path):
        # save the obj
        with open(path, 'wb') as fid:
            cPickle.dump(obj, fid)

    @staticmethod
    def obj_from_file(path):
        return cPickle.load(open(path, 'rb'))

    @staticmethod
    def cmp_classifiers(data_path, output_dir, dataset=None):
        data, labels, feature_names, vec = Learner.gen_instances('C:\Users\hfu\Documents\\flows\\normal\\March',
                                                                 data_path, simulate=False)
        data, feature_names, vec = Learner.feature_selection(data, labels, 200, vec,
                                                             feature_names=feature_names)
        cv_res = dict()
        clf, cv_r = Learner.train_tree(data, labels, cross_vali=True, feature_names=feature_names,
                                     tree_name='Fig_tree_sel_' + dataset, output_dir=output_dir)
        Learner.save2file(clf, classifier_dir + '\\' + 'tree_sel.pkl')
        cv_res['tree'] = cv_r

        clf, cv_r = Learner.train_bayes(data, labels, cross_vali=True)
        Learner.save2file(clf, classifier_dir + '\\' + 'bayes_sel.pkl')
        cv_res['bayes'] = cv_r

        clf, cv_r = Learner.train_logistic(data, labels, cross_vali=True)
        Learner.save2file(clf, classifier_dir + '\\' + 'logistic_sel.pkl')
        cv_res['logistic'] = cv_r

        clf, cv_r = Learner.train_SVM(data, labels, cross_vali=True)
        Learner.save2file(clf, classifier_dir + '\\' + 'svm_sel.pkl')
        cv_res['svm'] = cv_r

        json.dump(cv_res, codecs.open(output_dir + '/cv_res.json', 'w', encoding='utf-8'))


if __name__ == '__main__':
    logger = Utilities.set_logger('Learner')
    base_dir = 'C:\Users\hfu\Documents\\flows\CTU-13-Family\TCP-CC\\' #''C:\\Users\\hfu\\Documents\\flows\\CTU-13\\'
    #dataset_num = 'Neris' #'2'
    dataset = 'Neris' #''CTU-13-' + dataset_num + '\\'

    classifier_dir = base_dir + dataset
    #Learner.cmp_feature_selection(classifier_dir, classifier_dir, dataset=dataset)
    Learner.cmp_classifiers(classifier_dir, classifier_dir, dataset=dataset)

    """
    train = True
    learner = 'tree'
    if train:
        #"C:\Users\hfu\IdeaProjects\\recon\\test\\1",
         #                 "C:\Users\hfu\IdeaProjects\\recon\\test\\0")
        if learner == 'tree':
            classifier_dir = base_dir + dataset
            vocab_dir = base_dir + dataset
            simulate = False
            if simulate:
                classifier_dir = base_dir + 'CTU-13-1\\' + '1'
                vocab_dir = base_dir + 'CTU-13-1\\' + '1'
                data, labels, feature_names, vec = Learner.gen_instances(base_dir + 'CTU-13-1\\' + '1',
                                                                    None,
                                                                    output_dir=vocab_dir, simulate=simulate)
                data, feature_names = Learner.feature_selection(data, labels, 20000, vocab_dir, vec,
                                                                feature_names=feature_names)
                logger.info(data.shape)
                Learner.train_tree(data, labels, feature_names=feature_names, output_dir=classifier_dir,
                                   tree_name='Fig_tree_normal')
            else:
                data, labels, feature_names, vec = Learner.gen_instances('C:\Users\hfu\Documents\\flows\\normal\\March',
                                                                base_dir + dataset, simulate=False)
                data, feature_names, vec = Learner.feature_selection(data, labels, 200, vocab_dir, vec,
                                                                feature_names=feature_names)
                Learner.save2file(vec.vocabulary_, vocab_dir + '/' + "vocabulary.pkl")
                logger.info(data.shape)
                clf, cv = Learner.train_tree(data, labels, cross_vali=True, feature_names=feature_names,
                                   tree_name='Fig_tree_' + dataset_num)
                Learner.save2file(clf, classifier_dir + '\\' + 'classifier.pkl')

        elif learner == 'ocsvm':
            classifier_dir = base_dir + 'CTU-13-1\\' + '\\1'
            vocab_dir = base_dir + 'CTU-13-1\\' + '\\1'
            data, labels, feature_names, vec = Learner.gen_instances(vocab_dir,
                                                                base_dir + 'CTU-13-1\\' + '\\0\\CC',
                                                                output_dir=vocab_dir)
            Learner.ocsvm(data, labels, output_dir=classifier_dir)
    else:
        train_tree = 'Neris'
        test_data = 'Virut'
        test_normal = True

        if learner == 'tree':
            simulate = False
            classifier_dir = base_dir + train_tree
            vocab_dir = base_dir + dataset
            if simulate:
                classifier_dir = base_dir + 'CTU-13-1\\' + '\\1'
                vocab_dir = base_dir + 'CTU-13-1\\' + '\\1'
            learner_path = classifier_dir + '\\' + 'classifier.pkl'
        elif learner == 'ocsvm':
            classifier_dir = base_dir + 'CTU-13-1\\' + '\\1'
            vocab_dir = base_dir + 'CTU-13-1\\' + '\\1'
            learner_path = classifier_dir + '\\' + 'ocsvm.pkl'

        if test_normal == True:
            base_dir = 'C:\Users\hfu\Documents\\flows\\normal\\'
            test_data = 'April'
            data, labels = Learner.gen_instances(base_dir + test_data, '', to_vec=False)
        else:
            data, labels = Learner.gen_instances('', base_dir + test_data, to_vec=False)
        if learner == 'ocsvm':
            labels = [-1.] * len(labels)
        Learner.predict(Learner.obj_from_file(learner_path),
                            Learner.obj_from_file(vocab_dir + '\\' + 'vocabulary.pkl'), data, labels=labels)

    """