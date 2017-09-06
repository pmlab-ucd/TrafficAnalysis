from utils import Utilities
from CtuCCAnalyzer import CtuCCAnalyzer
#from pathos.multiprocessing import ProcessingPool as Pool
from threading import Thread
import simplejson
import os

class CtuAdAnalyzer:
    logger = Utilities.set_logger('CTU-Ad')

    @staticmethod
    def cv_result_table(base_dir):
        for model_name in ['bag', 'bag-ngram', 'tf', 'tf-ngram']:
            print '\\\\'
            model = dict()

            model_name = model_name + '_'

            for dataset in ['']:
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

                for dataset in ['']:
                    results = model[algorithm][dataset]
                    mean_conf = results['mean_conf_mat']
                    recall = str('{:.3%}'.format(mean_conf['recall'])).replace('%', '\%')
                    fp = str('{:.3%}'.format(mean_conf['fp_rate'])).replace('%', '\%')
                    precision = str('{:.3%}'.format(mean_conf['precision'])).replace('%', '\%')
                    f1 = str('{:.3%}'.format(mean_conf['f1score'])).replace('%', '\%')
                    mean_score = str('{:.3%}'.format(results['mean_scores'])).replace('%', '\%')
                    duration = str('{:.3}'.format(results['duration']))
                    print ' & ' + algorithm_name + ' & ' + duration + ' & ' + recall \
                          + ' & ' + fp + ' & ' + precision \
                          + ' & ' + f1 + ' & ' + mean_score + ' \\\\ '

    @staticmethod
    def cmp_model_cv(base_dir, normal_dir):
        """
        Cmp between bag-of-words, Tf-idf, bag-ngrams, Tf-ngrams
        :return:
        """
        classifier_dir = base_dir

        """
        Pool().map(CtuCCAnalyzer.cmp_algorithm_cv, [base_dir, base_dir, base_dir, base_dir],
                   [normal_dir, normal_dir, normal_dir, normal_dir],
                   [classifier_dir, classifier_dir, classifier_dir, classifier_dir],
                   ['bag_', 'bag-ngram_', 'tf_', 'tf-ngram_'])
        """
        threads = dict()
        for model_name in ['bag', 'bag-ngram', 'tf', 'tf-ngram']:
            threads[model_name] = Thread(target=CtuCCAnalyzer.cmp_algorithm_cv, args=(base_dir, normal_dir, classifier_dir,
                                               classifier_dir, model_name + '_'))
            threads[model_name].start()

        for model_name in threads:
            threads[model_name].join()

            #CtuAdAnalyzer.logger.info(model_name + "----------------------------------")
            #CtuCCAnalyzer.cmp_algorithm_cv(base_dir, normal_dir, classifier_dir, classifier_dir,
                                           #model_name=model_name + '_')

if __name__ == '__main__':
    base_dir = 'C:\Users\hfu\Documents\\flows\CTU-13-Family\Ad'
    normal_dir = 'C:\Users\hfu\Documents\\flows\\normal\\'
    CtuAdAnalyzer.cmp_model_cv(base_dir, normal_dir)
