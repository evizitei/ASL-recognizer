import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        cur_comp_count = self.min_n_components
        best_bic = 100000000
        best_comp_count = self.min_n_components
        while cur_comp_count <= self.max_n_components:
            try:
                candidate = GaussianHMM(n_components=cur_comp_count, covariance_type="diag", n_iter=1000,
                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = candidate.score(self.X, self.lengths)
                p_val = ( cur_comp_count ** 2 ) + ( 2 * cur_comp_count * sum(self.lengths) ) - 1
                bic = (-2 * logL) + (p_val * np.log(len(self.X)))
                if bic < best_bic:
                    best_bic = bic
                    best_comp_count = cur_comp_count
            except ValueError:
                cur_comp_count += 1
                continue
            cur_comp_count += 1
        return self.base_model(best_comp_count)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        cur_comp_count = self.min_n_components
        best_dic = -100000000
        best_comp_count = self.min_n_components
        while cur_comp_count <= self.max_n_components:
            try:
                candidate = GaussianHMM(n_components=cur_comp_count, covariance_type="diag", n_iter=1000,
                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = candidate.score(self.X, self.lengths)
                otherLogLs = []
                for w, (alt_X, alt_lengths) in self.hwords.items():
                    if w != self.this_word:
                        altLogL = candidate.score(alt_X, alt_lengths)
                        otherLogLs.append(altLogL)
                meanAltLog = np.mean(otherLogLs)
                dic = logL - meanAltLog
                if dic > best_dic:
                    best_dic = dic
                    best_comp_count = cur_comp_count
            except ValueError:
                cur_comp_count += 1
                continue
            cur_comp_count += 1
        return self.base_model(best_comp_count)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        cur_comp_count = self.min_n_components
        best_logL = -100000
        best_comp_count = self.min_n_components
        while cur_comp_count <= self.max_n_components:
            logLs = []
            split_count = 3
            if len(self.sequences) < 3:
                split_count = len(self.sequences)
            split_method = KFold(n_splits=split_count)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)

                if len(train_X) >= cur_comp_count:
                    try:
                        test_X, test_lengths = combine_sequences(cv_train_idx, self.sequences)
                        candidate = GaussianHMM(n_components=cur_comp_count, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                        logL = candidate.score(test_X, test_lengths)
                        logLs.append(logL)
                    except ValueError:
                        continue
            if len(logLs) > 0:
                avg_logL = np.mean(logLs)
                if avg_logL > best_logL:
                    best_logL = avg_logL
                    best_comp_count = cur_comp_count
            cur_comp_count += 1
        return self.base_model(best_comp_count)
