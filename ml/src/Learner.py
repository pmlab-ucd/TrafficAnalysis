import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
import pydotplus
import codecs
import cPickle
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import simplejson
import json
from utils import Utilities

import string
import random

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from nltk.stem.porter import PorterStemmer
import nltk

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([PorterStemmer().stem(w) for w in analyzer(doc)])


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([PorterStemmer().stem(w) for w in analyzer(doc)])


class LatexTableGenerator():
    @staticmethod
    def feature_tab(base_dir):
        # Open X and output the attribute amount
        for model_name in ['bag', 'bag-ngram', 'tf', 'tf-ngram']:
            if model_name == 'bag':
                model_n = 'Bag-of-word'
            elif model_name == 'tf':
                model_n = 'Tf-idf'
            elif model_name == 'bag-ngram':
                model_n = 'Bag-of-word-NGram'
            else:
                model_n = 'Tf-idf-NGram'
            model_name = model_name + '_'
            for dataset in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                if dataset != 'Neris':
                    model_n = ''
                line = model_n + ' & ' + dataset + '& '
                output_dir = base_dir + dataset
                X = Learner.obj_from_file(os.path.join(output_dir, model_name + "X.pkl"))
                line += str(X.shape[1]) + ' & 500 & '
                # print X.shape[1]
                feature_names = Learner.obj_from_file(os.path.join(output_dir, model_name + "feature_names_sel.pkl"))
                for i in range(1, 5):
                    feature_name = feature_names[i]
                    if len(feature_name) > 8:
                        feature_name = str(feature_name)[0:8]
                    line += feature_name + ', '
                line += ' ...\\\\ '
                print line

    @staticmethod
    def parse_zero_day_res(base_dir):
        for model_name in ['bag', 'bag-ngram', 'tf', 'tf-ngram']:
            print model_name + '__________________________'
            model = dict()

            model_name = model_name + '_'
            with open(os.path.join(base_dir, model_name + 'pred_res.json'), "rb") as fin:
                pred_res = simplejson.load(fin)
                for algorithm in ['tree', 'bayes', 'logistic', 'svm', 'ocsvm']:
                    print '\\\\'
                    if algorithm == 'tree':
                        algorithm_name = 'Decision Tree'
                    elif algorithm == 'bayes':
                        algorithm_name = 'Naive Bayes'
                    elif algorithm == 'logistic':
                        algorithm_name = 'Logistic Regression'
                    elif algorithm == 'svm':
                        algorithm_name = 'SVM'
                    else:
                        algorithm_name = 'OCSVM'

                    for src in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                        if src != 'Neris':
                            algorithm_name = ''
                        results = pred_res[algorithm][src]
                        line = algorithm_name + ' & '
                        line += src + ' & '
                        for tgt in ['Neris', 'Murlo', 'Virut', 'Sogou', 'April']:
                            res = str('{:.3%}'.format(results[tgt])).replace('%', '\%')
                            line += res
                            line += ' & '
                        line = line[:-2]
                        line += '\\\\'
                        print line

    @staticmethod
    def cv_result_table(base_dir):
        for model_name in ['bag', 'bag-ngram', 'tf', 'tf-ngram']:
            print model_name + '__________________________'
            model = dict()

            model_name = model_name + '_'

            for dataset in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                output_dir = base_dir + dataset

                with open(os.path.join(output_dir, model_name + 'cv_res_sel.json'), "rb") as fin:
                    cv_res = simplejson.load(fin)
                    # print cv_res
                    for algorithm in cv_res:
                        if algorithm not in model:
                            model[algorithm] = dict()
                        results = cv_res[algorithm]
                        model[algorithm][dataset] = results
                        # print(algorithm + ': ' + str(results['duration']))
                        # print('mean scores:' + str(results['mean_scores']))
                        # print('mean_conf:' + str(results['mean_conf_mat']))

            for algorithm in ['tree', 'bayes', 'logistic', 'svm', 'ocsvm']:
                print '\\\\'
                if algorithm == 'tree':
                    algorithm_name = 'Decision Tree'
                elif algorithm == 'bayes':
                    algorithm_name = 'Naive Bayes'
                elif algorithm == 'logistic':
                    algorithm_name = 'Logistic Regreesion'
                elif algorithm == 'svm':
                    algorithm_name = 'SVM'
                else:
                    algorithm_name = 'OCSVM'

                for dataset in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                    if dataset != 'Neris':
                        algorithm_name = ''
                    results = model[algorithm][dataset]
                    mean_conf = results['mean_conf_mat']
                    recall = str('{:.3%}'.format(mean_conf['recall'])).replace('%', '\%')
                    fp = str('{:.3%}'.format(mean_conf['fp_rate'])).replace('%', '\%')
                    precision = str('{:.3%}'.format(mean_conf['precision'])).replace('%', '\%')
                    f1 = str('{:.3%}'.format(mean_conf['f1score'])).replace('%', '\%')
                    mean_score = str('{:.3%}'.format(results['mean_scores'])).replace('%', '\%')
                    print algorithm_name + ' & ' + dataset + ' & ' + str(results['duration']) + ' & ' + recall \
                          + ' & ' + fp + ' & ' + precision \
                          + ' & ' + f1 + ' & ' + mean_score + ' \\\\ '


class Learner:
    global logger

    class LabelledDocs:
        def stem_tokens(tokens, stemmer):
            stemmed = []
            for item in tokens:
                stemmed.append(stemmer.stem(item))
            return stemmed

        def tokenize(self, text):
            vectorizer = CountVectorizer(analyzer='word')
            vectorizer.fit_transform([text])
            tokens = vectorizer.get_feature_names()
            # stems = self.stem_tokens(tokens, stemmer)
            return tokens

        def __init__(self, doc, label, char_wb=False):
            self.doc = doc
            self.label = label
            tokens = self.tokenize(doc)
            if char_wb:
                self.doc = ''.join(tokens)
            else:
                self.doc = ' '.join(tokens)

    @staticmethod
    def dir2jsons(json_dir):
        jsons = []
        if json_dir is None:
            return jsons
        for root, dirs, files in os.walk(json_dir, topdown=False):
            for filename in files:
                if '201' in filename and re.search('json$', filename):
                    with open(os.path.join(root, filename), "rb") as fin:
                        try:
                            jsons.append(simplejson.load(fin))
                        except Exception as e:
                            pass
                            # Utilities.logger.error(e)
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
            logger.info('i: ' + vocab[i] + ' ' + str(i))
            if len(vocab[i]) < 6 or vocab[i] in examined:
                continue
            for j in range(i + 1, len(vocab)):
                # logger.info('j: ' + vocab[j] + ' ' + str(j))
                if len(vocab[j]) < 6:
                    examined.append(vocab[j])
                    continue
                if vocab[i] in vocab[j] or vocab[j] in vocab[i]:  # Learner.same_prefix(vocab[i], vocab[j]):
                    # logger.info('Found ' + vocab[i] + ' ' + vocab[j] + ' ' + str(i))
                    examined.append(vocab[j])
                    for doc in docs:
                        if vocab[j] in doc.doc:
                            doc.doc = str(doc.doc).replace(vocab[j], vocab[i])
        instances = []
        labels = []
        for doc in docs:
            instances.append(doc.doc)
            labels.append(doc.label)
        vectorizer = StemmedCountVectorizer(analyzer="word",
                                            tokenizer=None,
                                            preprocessor=None,
                                            stop_words=None)
        train_data = vectorizer.fit_transform(instances)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        # train_data = train_data.toarray()
        logger.info(train_data.shape)
        return train_data, labels

    @staticmethod
    def gen_instances(pos_json_dir, neg_json_dir, simulate=False, char_wb=False):
        pos_jsons = Learner.dir2jsons(pos_json_dir)
        neg_jsons = Learner.dir2jsons(neg_json_dir)
        logger.info('lenPos: ' + str(len(pos_jsons)))
        logger.info('lenNeg: ' + str(len(neg_jsons)))
        docs = Learner.gen_docs(pos_jsons, 1, char_wb)
        docs = docs + (Learner.gen_docs(neg_jsons, -1, char_wb))
        if simulate:
            if len(neg_jsons) == 0:
                docs = docs + Learner.simulate_flows(len(pos_jsons), 0)
        instances = []
        labels = []
        for doc in docs:
            instances.append(doc.doc)
            labels.append(doc.label)

        return instances, np.array(labels)

    @staticmethod
    def gen_X_matrix(instances, vec=None, tf=False, ngrams_range=None):
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        if vec is not None:
            train_data = vec.transform(instances)
            vocab = vec.get_feature_names()
            return train_data, vocab, vec
        if not tf:
            if ngrams_range is None:
                vectorizer = StemmedCountVectorizer(analyzer="word",
                                                    tokenizer=None,
                                                    preprocessor=None,
                                                    stop_words=['http'])
            else:
                vectorizer = StemmedCountVectorizer(analyzer='char_wb',
                                                    tokenizer=None,
                                                    preprocessor=None,
                                                    stop_words=['http'],
                                                    ngram_range=ngrams_range)
        else:
            if ngrams_range is None:
                vectorizer = StemmedTfidfVectorizer(analyzer="word",
                                                    tokenizer=None,
                                                    preprocessor=None,
                                                    stop_words=['http'])
            else:
                vectorizer = StemmedTfidfVectorizer(analyzer='char_wb',
                                                    tokenizer=None,
                                                    preprocessor=None,
                                                    stop_words=None,
                                                    ngram_range=ngrams_range)
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
        # train_data, labels = Learner.feature_filter_by_prefix(vocab, docs)

        return train_data, vocab, vectorizer

    @staticmethod
    def ocsvm(train_data, labels, cross_vali=True):
        nu = float(np.count_nonzero(labels == -1)) / len(labels)
        clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)
        results = None
        if cross_vali:
            results = Learner.cross_validation(clf, train_data, labels)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('OCSVM: ' + str(results['duration']))
            logger.info('mean scores:' + str(results['mean_scores']))
            logger.info('mean_conf:' + str(results['mean_conf_mat']))

        clf.fit(train_data)

        return clf, results

    @staticmethod
    def train_bayes(train_data, labels, cross_vali=True):
        clf = BernoulliNB()
        results = None
        if cross_vali:
            results = Learner.cross_validation(clf, train_data, labels)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Bayes: ' + str(results['duration']))
            logger.info('mean scores:' + str(results['mean_scores']))
            logger.info('mean_conf:' + str(results['mean_conf_mat']))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        return clf, results

    @staticmethod
    def class_report(conf_mat):
        tp, fp, fn, tn = conf_mat.flatten()
        measures = {'accuracy': (tp + tn) / (tp + fp + fn + tn), 'fp_rate': fp / (tn + fp), 'recall': tp / (tp + fn),
                    'precision': tp / (tp + fp), 'f1score': 2 * tp / (2 * tp + fp + fn)}
        # measures['tn_rate'] = tn / (tn + fp)  # (true negative rate)
        return measures

    @staticmethod
    def cross_validation(clf, data, labels, scoring='f1', n_fold=5):
        X = data
        y = np.array(labels)
        ''' Run x-validation and return scores, averaged confusion matrix, and df with false positives and negatives '''
        t0 = time()
        results = dict()
        # cv = KFold(n_splits=5, shuffle=True)

        # I generate a KFold in order to make cross validation
        shuffle = True
        kf = StratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=42)
        scores = []
        conf_mat = np.zeros((2, 2))  # Binary classification

        # I start the cross validation
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            result = dict()
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # I train the classifier
            clf.fit(X_train, y_train)

            # I make the predictions
            predicted = clf.predict(X_test)
            y_plabs = np.squeeze(predicted)
            if hasattr(clf, 'predict_proba'):
                y_pprobs = clf.predict_proba(X_test)  # Predicted probabilitie
                result['roc'] = metrics.roc_auc_score(y_test, y_pprobs[:, 1])
            else:  # for SVM
                y_decision = clf.decision_function(X_test)
                try:
                    result['roc'] = metrics.roc_auc_score(y_test, y_decision[:, 1])
                except:  # OCSVM
                    result['roc'] = metrics.roc_auc_score(y_test, y_decision)
            # metrics.roc_curve(y_test, y_pprobs[:, 1])
            scores.append(result['roc'])

            # Learner.perf_measure(predicted, y_test)

            # I obtain the accuracy of this fold
            # ac = accuracy_score(predicted, y_test)

            # I obtain the confusion matrix
            confusion = metrics.confusion_matrix(y_test, predicted)
            conf_mat += confusion
            result['conf_mat'] = confusion.tolist()

            # Collect indices of false positive and negatives, effective only shuffle=False, or backup the original data
            if not shuffle:
                fp_i = np.where((y_plabs == 1) & (y_test == -1))[0]
                fn_i = np.where((y_plabs == -1) & (y_test == 1))[0]
                result['fp_item'] = test_index[fp_i]
                result['fn_item'] = test_index[fn_i]
            results['fold_' + str(fold)] = result

        # cv_res = cross_val_score(clf, data, labels, cv=cv, scoring='f1').tolist()
        # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
        # separators=(',', ':'), sort_keys=True, indent=4)
        duration = time() - t0
        results['duration'] = duration
        # results['cv_res'] = cv_res
        # results['cv_res_mean'] = sum(cv_res) / n_splits

        # print "\nMean score: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2)
        results['mean_scores'] = np.mean(scores)
        results['std_scores'] = np.std(scores)
        conf_mat /= n_fold
        # print "Mean CM: \n", conf_mat

        # print "\nMean classification measures: \n"
        results['mean_conf_mat'] = Learner.class_report(conf_mat)
        # return scores, conf_mat, {'fp': sorted(false_pos), 'fn': sorted(false_neg)}
        return results

    @staticmethod
    def train_SVM(train_data, labels, cross_vali=True):
        clf = svm.SVC(class_weight='balanced', probability=True)
        results = None
        if cross_vali == True:
            results = Learner.cross_validation(clf, train_data, labels)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('SVM: ' + str(results['duration']))
            logger.info('mean scores:' + str(results['mean_scores']))
            logger.info('mean_conf:' + str(results['mean_conf_mat']))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        return clf, results

    @staticmethod
    def train_logistic(train_data, labels, cross_vali=True):
        clf = LogisticRegression(class_weight='balanced')
        results = None
        if cross_vali == True:
            results = Learner.cross_validation(clf, train_data, labels)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Logistic: ' + str(results['duration']))
            logger.info('mean scores:' + str(results['mean_scores']))
            logger.info('mean_conf:' + str(results['mean_conf_mat']))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)

        return clf, results

    @staticmethod
    def train_tree(train_data, labels, cross_vali=True, output_dir=os.curdir, tree_name='tree'):
        clf = DecisionTreeClassifier(class_weight='balanced')
        results = None
        if cross_vali == True:
            results = Learner.cross_validation(clf, train_data, labels)
            # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'),
            # separators=(',', ':'), sort_keys=True, indent=4)
            logger.info('Tree: ' + str(results['duration']))
            logger.info('mean scores:' + str(results['mean_scores']))
            logger.info('mean_conf:' + str(results['mean_conf_mat']))

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        clf = clf.fit(train_data, labels)
        """
        tree.export_graphviz(clf, out_file=output_dir + '/' + tree_name + '.dot',
                             feature_names=feature_names,
                             label='root', impurity=False, special_characters=True)  # , max_depth=5)
        dotfile = open(output_dir + '/' + tree_name + '.dot', 'r')
        graph = pydotplus.graph_from_dot_data(dotfile.read())
        graph.write_pdf(output_dir + '/' + tree_name + '.pdf')
        dotfile.close()
        """
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
    def gen_docs(jsons, label, char_wb=False):
        docs = []
        for flow in jsons:
            label = label  # flow['label']
            line = ''
            line += flow['domain']
            line += flow['uri']
            try:
                docs.append(Learner.LabelledDocs(line, label, char_wb=char_wb))
            except:
                print line
        return docs

    @staticmethod
    def predict(model, vec, instances, labels=None, src_name='', model_name=''):
        # loaded_vec = CountVectorizer(decode_error="replace", vocabulary=voc)
        data = vec.transform(instances)
        y_1 = model.predict(data)

        # logger.info(y_1)
        if labels is not None:
            return accuracy_score(labels, y_1)

    @staticmethod
    def feature_selection(X, y, k, count_vectorizer, instances, tf=False, ngram_range=None):
        ch2 = SelectKBest(chi2, k=k)
        X_new = ch2.fit_transform(X, y)
        feature_names = count_vectorizer.get_feature_names()
        if feature_names != None:
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]
        '''
        dict = np.asarray(count_vectorizer.get_feature_names())[ch2.get_support()]
        if tf:
            if ngram_range is not None:
                count_vectorizer = StemmedTfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, vocabulary=dict)
            else:
                count_vectorizer = StemmedTfidfVectorizer(analyzer='char_wb', vocabulary=dict)
        else:
            if ngram_range is not None:
                count_vectorizer = StemmedCountVectorizer(analyzer='word', vocabulary=dict, ngram_range=ngram_range)
            else:
                count_vectorizer = StemmedCountVectorizer(analyzer="word", vocabulary=dict)
        X_new = count_vectorizer.fit_transform(instances)
        # cPickle.dump(count_vectorizer.vocabulary, open(output_dir + '/' + "vocabulary.pkl", "wb"))
        '''
        return X_new, feature_names, ch2

    @staticmethod
    def pipe_feature_selection(X, y):
        clf = Pipeline([
            ('feature_selection', SelectKBest(chi2, k=2).fit_transform(X, y)),
            ('classification', RandomForestClassifier())
        ])
        clf.fit(X, y)

    @staticmethod
    def cmp_feature_selection(data_path, output_dir, dataset=None):
        classifier_dir = base_dir + dataset
        instances, labels = Learner.gen_instances(os.path.join(normal_dir, 'March'),
                                                  data_path, simulate=False)
        data, feature_names, vec = Learner.gen_X_matrix(instances)
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
        data, feature_names, vec = Learner.feature_selection(data, labels, 200, vec, instances)

        Learner.save2file(vec.vocabulary, output_dir + '/' + "vocabulary_sel.pkl")
        logger.info(data.shape)
        clf, cv = Learner.train_tree(data, labels, cross_vali=True,
                                     tree_name='Fig_tree_sel_' + dataset, output_dir=output_dir)
        Learner.save2file(clf, classifier_dir + '\\' + 'classifier_sel.pkl')

        clf_info = Learner.tree_info(clf)
        clf_info['cv'] = cv

        json.dump(clf_info, codecs.open(output_dir + '/tree_info_sel.json', 'w', encoding='utf-8'))


        # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'))
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
    def cmp_model_cv(base_dir):
        """
        Cmp between bag-of-words, Tf-idf, bag-ngrams, Tf-ngrams
        :return:
        """
        for model_name in ['bag']: # 'bag', 'bag', 'bag-ngram',
            logger.info(model_name + "----------------------------------")
            for dataset in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                classifier_dir = base_dir + dataset
                Learner.cmp_algorithm_cv(classifier_dir, classifier_dir, dataset=dataset, model_name=model_name + '_')

    @staticmethod
    def cmp_algorithm_cv(data_path, output_dir, dataset=None, model_name=''):
        char_wb = False
        if 'tf' in model_name:
            tf = True
        else:
            tf = False
        if 'ngram' in model_name:
            ngram = (2, 15)
            # char_wb = True
        else:
            ngram = None

        classifier_dir = base_dir + dataset
        if os.path.exists(os.path.join(output_dir, model_name + "vec_sel.pkl")):
            X = Learner.obj_from_file(os.path.join(output_dir, model_name + "X_sel.pkl"))
            y = Learner.obj_from_file(os.path.join(output_dir, model_name + "y_sel.pkl"))
        else:
            instances, y = Learner.gen_instances(os.path.join(normal_dir, 'March'),
                                                 data_path, char_wb=char_wb, simulate=False)
            X, feature_names, vec = Learner.gen_X_matrix(instances, tf=tf, ngrams_range=ngram)

            Learner.save2file(X, os.path.join(output_dir, model_name + "X.pkl"))
            Learner.save2file(y, os.path.join(output_dir, model_name + "y.pkl"))
            Learner.save2file(vec, os.path.join(output_dir, model_name + "vec.pkl"))
            Learner.save2file(feature_names, os.path.join(output_dir, model_name + "feature_names.pkl"))
            X, feature_names, vec = Learner.feature_selection(X, y, 500, vec, instances, tf=tf, ngram_range=ngram)
            Learner.save2file(X, os.path.join(output_dir, model_name + "X_sel.pkl"))
            Learner.save2file(y, os.path.join(output_dir, model_name + "y_sel.pkl"))
            Learner.save2file(vec, os.path.join(output_dir, model_name + "vec_sel.pkl"))
            Learner.save2file(feature_names, os.path.join(output_dir, model_name + "feature_names_sel.pkl"))
        """
        cv_res = dict()
        clf, cv_r = Learner.train_tree(X, y, cross_vali=True, tree_name='Fig_tree_sel_' + dataset,
                                       output_dir=output_dir)
        Learner.save2file(clf, classifier_dir + '\\' + model_name + 'tree_sel.pkl')
        cv_res['tree'] = cv_r

        clf, cv_r = Learner.train_bayes(X, y, cross_vali=True)
        Learner.save2file(clf, classifier_dir + '\\' + model_name + 'bayes_sel.pkl')
        cv_res['bayes'] = cv_r

        clf, cv_r = Learner.train_logistic(X, y, cross_vali=True)
        Learner.save2file(clf, classifier_dir + '\\' + model_name + 'logistic_sel.pkl')
        cv_res['logistic'] = cv_r

        clf, cv_r = Learner.train_SVM(X, y, cross_vali=True)
        Learner.save2file(clf, classifier_dir + '\\' + model_name + 'svm_sel.pkl')
        cv_res['svm'] = cv_r

        clf, cv_r = Learner.ocsvm(X, y, cross_vali=True)
        Learner.save2file(clf, classifier_dir + '\\' + model_name + 'ocsvm_sel.pkl')
        cv_res['ocsvm'] = cv_r

        json.dump(cv_res, codecs.open(os.path.join(output_dir, model_name + 'cv_res_sel.json'), 'w', encoding='utf-8'))
        """

    @staticmethod
    def zero_day_helper(base_dir, src_name, model_name, algorithm, target_name, normal_dir=None):
        vec_dir = os.path.join(base_dir, src_name)
        model_path = os.path.join(vec_dir, model_name + algorithm + '_sel.pkl')
        target_path = os.path.join(base_dir, target_name)
        if normal_dir is None:
            data, labels = Learner.gen_instances('', target_path)
        else:
            data, labels = Learner.gen_instances(os.path.join(normal_dir, target_name), '')
        vec = Learner.obj_from_file(os.path.join(vec_dir, model_name + 'vec.pkl'))
        vec_sel = Learner.obj_from_file(os.path.join(vec_dir, model_name + 'vec_sel.pkl'))
        data, vocab, vec = Learner.gen_X_matrix(data, vec=vec)
        return Learner.predict(Learner.obj_from_file(model_path),
                               vec_sel, data, labels=labels,
                               src_name=src_name, model_name=model_name)

    @staticmethod
    def zero_day_sub(base_dir, model_name, output_dir):
        if os.path.exists(os.path.join(output_dir, model_name + 'pred_res.json')):
            return
        results = dict()
        for algorithm in ['tree', 'bayes', 'logistic', 'svm', 'ocsvm']:
            for src_name in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                for target_name in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                    res = Learner.zero_day_helper(base_dir, src_name, model_name, algorithm, target_name)
                    if algorithm not in results:
                        results[algorithm] = dict()
                    if src_name not in results[algorithm]:
                        results[algorithm][src_name] = dict()
                    results[algorithm][src_name][target_name] = res
                    # name = src_name + '_' + model_name + '_' + target_name
                    # logger.info(name + ':' + str(res))
                target_name = 'April'
                res = Learner.zero_day_helper(base_dir, src_name, model_name, algorithm, target_name,
                                              normal_dir=normal_dir)
                # name = src_name + '_' + model_name + '_' + target_name
                # logger.info(name + ':' + str(res))
                results[algorithm][src_name][target_name] = res
        json.dump(results, codecs.open(os.path.join(output_dir, model_name + 'pred_res.json'), 'w', encoding='utf-8'))

        for algorithm in ['tree', 'bayes', 'logistic', 'svm', 'ocsvm']:
            for src_name in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                output = ''
                for target_name in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                    output = output + str(results[algorithm][src_name][target_name] * 100) + '\%' + ' & '
                logger.info(algorithm + ' & ' + src_name + ' & ' + output)

    @staticmethod
    def zero_day(base_dir):
        for model_name in ['bag']: #['bag', 'bag-ngram', 'tf', 'tf-ngram']:
            Learner.zero_day_sub(base_dir, model_name + '_', base_dir)


if __name__ == '__main__':
    logger = Utilities.set_logger('Learner')
    base_dir = 'E:\\flows\CTU-13-Family\TCP-CC\\'  # ''C:\\Users\\hfu\\Documents\\flows\\CTU-13\\'
    normal_dir = 'E:\\flows\\normal\\'
    # dataset_num = 'Neris' #'2'

    # Learner.cmp_feature_selection(classifier_dir, classifier_dir, dataset=dataset)

    Learner.cmp_model_cv(base_dir)

    Learner.zero_day(base_dir)

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
