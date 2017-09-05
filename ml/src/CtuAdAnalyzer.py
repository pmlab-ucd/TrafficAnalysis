from utils import Utilities
from CtuCCAnalyzer import CtuCCAnalyzer
#from pathos.multiprocessing import ProcessingPool as Pool
from threading import Thread

class CtuAdAnalyzer:
    logger = Utilities.set_logger('CTU-Ad')

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
