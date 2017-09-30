import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for i in range(test_set.num_items):
        word = test_set.wordlist[i]
        seqs = test_set.get_item_sequences(i)
        X = test_set.get_item_Xlengths(i)
        p_hash = {}
        best_score = -1000000000
        best_word = None
        for m_key in models:
            try:
                model = models[m_key]
                logL = model.score(X[0], X[1])
                p_hash[m_key] = logL
                if logL > best_score:
                    best_score = logL
                    best_word = m_key
            except ValueError:
                continue
        probabilities.append(p_hash)
        guesses.append(best_word)
    return (probabilities, guesses)
