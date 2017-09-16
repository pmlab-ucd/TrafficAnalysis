from Learner import Learner
from utils import Utilities
import json
import os
import simplejson
import codecs
# from pathos.multiprocessing import ProcessingPool as Pool
# import pathos.multiprocessing
# import multiprocessing
from threading import Thread
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dt
from PacpHandler import PcapHandler

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
                output_dir = os.path.join(base_dir, dataset)

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
                    algorithm_name = 'Logistic Regression'
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
                    duration = str('{:.3}'.format(results['duration']))
                    print algorithm_name + ' & ' + dataset + ' & ' + duration + ' & ' + recall \
                          + ' & ' + fp + ' & ' + precision \
                          + ' & ' + f1 + ' & ' + mean_score + ' \\\\ '

    @staticmethod
    def event_duration(out_dir):
        timestamps = []
        for root, dirs, files in os.walk(out_dir, topdown=True):
            for name in files:
                # print(os.path.join(root, name))
                if str(name).endswith('.pcap'):
                    pcap_path = os.path.join(root, name)
                    timestamps.append(PcapHandler.duration_pcap(pcap_path))
        print timestamps
        '''
        df = pd.read_csv('data.csv')
        df.amin = pd.to_datetime(df.amin).astype(datetime)
        df.amax = pd.to_datetime(df.amax).astype(datetime)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = ax.xaxis_date()
        ax = plt.hlines(df.index, dt.date2num(df.amin), dt.date2num(df.amax))
        '''


class CtuCCAnalyzer:
    logger = Utilities.set_logger('CTU-13-CC')

    @staticmethod
    def cmp_feature_selection(base_dir, normal_dir, data_path, output_dir, dataset=None):
        classifier_dir = base_dir + dataset
        instances, labels = Learner.gen_instances(os.path.join(normal_dir, 'March'),
                                                  data_path, simulate=False)
        data, feature_names, vec = Learner.gen_X_matrix(instances)
        back = [data, labels, feature_names, vec]

        Learner.save2file(vec.vocabulary_, output_dir + '/' + "vocabulary.pkl")
        CtuCCAnalyzer.logger.info(data.shape)
        clf, cv = Learner.train_tree(data, labels, cross_vali=True,
                                     tree_name='Fig_tree_' + dataset, output_dir=output_dir)
        Learner.save2file(clf, classifier_dir + '\\' + 'classifier.pkl')

        clf_info = Learner.tree_info(clf)
        clf_info['cv'] = cv

        simplejson.dump(clf_info, codecs.open(output_dir + '/tree_info.json', 'w', encoding='utf-8'))

        data, labels, feature_names, vec = back
        data, feature_names, vec = Learner.feature_selection(data, labels, 200, vec, instances)

        Learner.save2file(vec.vocabulary, output_dir + '/' + "vocabulary_sel.pkl")
        CtuCCAnalyzer.logger.info(data.shape)
        clf, cv = Learner.train_tree(data, labels, cross_vali=True,
                                     tree_name='Fig_tree_sel_' + dataset, output_dir=output_dir)
        Learner.save2file(clf, classifier_dir + '\\' + 'classifier_sel.pkl')

        clf_info = Learner.tree_info(clf)
        clf_info['cv'] = cv

        json.dump(clf_info, codecs.open(output_dir + '/tree_info_sel.json', 'w', encoding='utf-8'))
        # simplejson.dump(results.tolist(), codecs.open(output_dir + '/cv.json', 'w', encoding='utf-8'))
        # separators=(',', ':'), sort_keys=True, indent=4)

    @staticmethod
    def cmp_model_cv(base_dir, normal_dir):
        """
        Cmp between bag-of-words, Tf-idf, bag-ngrams, Tf-ngrams
        :return:
        """
        for model_name in ['bag']:  # 'bag-ngram', 'tf', 'tf-ngram']:
            CtuCCAnalyzer.logger.info(model_name + "----------------------------------")
            for dataset in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                classifier_dir = base_dir + dataset
                CtuCCAnalyzer.cmp_algorithm_cv(base_dir, normal_dir, classifier_dir, classifier_dir,
                                               dataset=dataset, model_name=model_name + '_')

    @staticmethod
    def train_and_save(X, y, model_name, classifier_dir):
        outfile = os.path.join(classifier_dir, model_name + 'cv_res_sel.json')
        cv_res = dict()
        results = dict()
        thread1 = Thread(target=Learner.train_classifier, args=(Learner.train_tree, X, y, True, results, 'tree'))
        thread2 = Thread(target=Learner.train_classifier, args=(Learner.train_bayes, X, y, True, results, 'bayes'))
        thread3 = Thread(target=Learner.train_classifier,
                         args=(Learner.train_logistic, X, y, True, results, 'logistic'))
        thread4 = Thread(target=Learner.train_classifier, args=(Learner.train_SVM, X, y, True, results, 'svm'))
        thread5 = Thread(target=Learner.train_classifier, args=(Learner.ocsvm, X, y, True, results, 'ocsvm'))

        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        thread5.start()

        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()
        thread5.join()

        clf_tree, cv_res['tree'] = results['tree']
        clf_bayes, cv_res['bayes'] = results['bayes']
        clf_logistic, cv_res['logistic'] = results['logistic']
        clf_svm, cv_res['svm'] = results['svm']
        clf_ocsvm, cv_res['ocsvm'] = results['ocsvm']
        Learner.save2file(clf_tree, os.path.join(classifier_dir, model_name + 'tree_sel.pkl'))
        Learner.save2file(clf_bayes, os.path.join(classifier_dir, model_name + 'bayes_sel.pkl'))
        Learner.save2file(clf_logistic, os.path.join(classifier_dir, model_name + 'logistic_sel.pkl'))
        Learner.save2file(clf_svm, os.path.join(classifier_dir, model_name + 'svm_sel.pkl'))
        Learner.save2file(clf_ocsvm, os.path.join(classifier_dir, model_name + 'ocsvm_sel.pkl'))
        CtuCCAnalyzer.logger.info('Threads Done! Saving cv_res...')
        json.dump(cv_res, codecs.open(outfile, 'w', encoding='utf-8'))
        """

        result1, result2, result3, result4, result5 = Pool().map(Learner.train_classifier,
                            [Learner.train_tree, Learner.train_bayes, Learner.train_logistic, Learner.train_SVM, Learner.ocsvm],
                            [X, X, X, X, X], [y, y, y, y, y], [True, True, True, True, True])

        clf_tree, cv_res['tree'] = result1
        clf_bayes, cv_res['bayes'] = result2
        clf_logistic, cv_res['logistic'] = result3
        clf_svm, cv_res['svm'] = result4
        clf_ocsvm, cv_res['ocsvm'] = result5
        Learner.save2file(clf_tree, os.path.join(classifier_dir, model_name + 'tree_sel.pkl'))
        Learner.save2file(clf_bayes, os.path.join(classifier_dir, model_name + 'bayes_sel.pkl'))
        Learner.save2file(clf_logistic, os.path.join(classifier_dir, model_name + 'logistic_sel.pkl'))
        Learner.save2file(clf_svm, os.path.join(classifier_dir, model_name + 'svm_sel.pkl'))
        Learner.save2file(clf_ocsvm, os.path.join(classifier_dir, model_name + 'ocsvm_sel.pkl'))
        json.dump(cv_res,
                  codecs.open(os.path.join(classifier_dir, model_name + 'cv_res_sel.json'), 'w', encoding='utf-8'))
        '''

        result1 = Pool().map(Learner.train_tree, [X,], [y], [True])
        result2 = Pool().map(Learner.train_bayes, [X], [y], [True])
        result3 = Pool().map(Learner.train_logistic, [X], [y], [True])
        result4 = Pool().map(Learner.train_SVM, [X], [y], [True])
        result5 = Pool().map(Learner.ocsvm, [X], [y], [True])
        '''
        """

    @staticmethod
    def cmp_algorithm_cv(base_dir, normal_dir, data_path, output_dir, model_name='', dataset=''):
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
        outfile = os.path.join(classifier_dir, model_name + 'cv_res_sel.json')
        if os.path.exists(outfile):
            return

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
        CtuCCAnalyzer.train_and_save(X, y, model_name, classifier_dir)

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
    def zero_day_sub(base_dir, normal_dir, model_name, output_dir):
        if os.path.exists(os.path.join(output_dir, model_name + 'pred_res.json')):
            return
        results = dict()
        for algorithm in ['tree', 'bayes', 'logistic', 'svm', 'ocsvm']:
            for src_name in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                for target_name in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                    res = CtuCCAnalyzer.zero_day_helper(base_dir, src_name, model_name, algorithm, target_name)
                    if algorithm not in results:
                        results[algorithm] = dict()
                    if src_name not in results[algorithm]:
                        results[algorithm][src_name] = dict()
                    results[algorithm][src_name][target_name] = res
                    # name = src_name + '_' + model_name + '_' + target_name
                    # CtuCCAnalyzer.logger.info(name + ':' + str(res))
                target_name = 'April'
                res = CtuCCAnalyzer.zero_day_helper(base_dir, src_name, model_name, algorithm, target_name,
                                                    normal_dir=normal_dir)
                # name = src_name + '_' + model_name + '_' + target_name
                # CtuCCAnalyzer.logger.info(name + ':' + str(res))
                results[algorithm][src_name][target_name] = res
        json.dump(results, codecs.open(os.path.join(output_dir, model_name + 'pred_res.json'), 'w', encoding='utf-8'))

        for algorithm in ['tree', 'bayes', 'logistic', 'svm', 'ocsvm']:
            for src_name in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                output = ''
                for target_name in ['Neris', 'Murlo', 'Virut', 'Sogou']:
                    output = output + str(results[algorithm][src_name][target_name] * 100) + '\%' + ' & '
                CtuCCAnalyzer.logger.info(algorithm + ' & ' + src_name + ' & ' + output)

    @staticmethod
    def zero_day(base_dir, normal_dir):
        for model_name in ['bag']:  # ['bag', 'bag-ngram', 'tf', 'tf-ngram']:
            CtuCCAnalyzer.zero_day_sub(base_dir, normal_dir, model_name + '_', base_dir)


if __name__ == '__main__':
    # global base_dir, normal_dir
    base_dir = 'E:\\flows\CTU-13-Family\TCP-CC\\'  # ''C:\\Users\\hfu\\Documents\\flows\\CTU-13\\'
    normal_dir = 'E:\\flows\\normal\\'
    # dataset_num = 'Neris' #'2'
    CtuCCAnalyzer.cmp_model_cv(base_dir, normal_dir)
    CtuCCAnalyzer.zero_day(base_dir, normal_dir)

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
                CtuCCAnalyzer.logger.info(data.shape)
                Learner.train_tree(data, labels, feature_names=feature_names, output_dir=classifier_dir,
                                   tree_name='Fig_tree_normal')
            else:
                data, labels, feature_names, vec = Learner.gen_instances('C:\Users\hfu\Documents\\flows\\normal\\March',
                                                                base_dir + dataset, simulate=False)
                data, feature_names, vec = Learner.feature_selection(data, labels, 200, vocab_dir, vec,
                                                                feature_names=feature_names)
                Learner.save2file(vec.vocabulary_, vocab_dir + '/' + "vocabulary.pkl")
                CtuCCAnalyzer.logger.info(data.shape)
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
