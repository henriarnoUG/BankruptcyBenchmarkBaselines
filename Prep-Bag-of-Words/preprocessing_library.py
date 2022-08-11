# IMPORTS
# ______________________________________________________________________________________________________________________
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import pandas as pd
import os
import gc
import pickle
import gensim
import time


# LIBRARY - DOCUMENT-LEVEL FUNCTIONS
# ______________________________________________________________________________________________________________________
def tokenize_lemmatize(text):
    """
    :param text: input string to process
    :return: list of lemmatized tokens
    """

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # convert to lower
    text = str(text).lower()
    # tokenize
    text = nltk.word_tokenize(text)
    # lemmatize
    text = list(map(lemmatizer.lemmatize, text))
    # final check if text is in list format
    if isinstance(text, list):
        pass
    else:
        text = nltk.word_tokenize(str(text))
    # return result
    return text


def tokenize(text):
    """
    :param text: input string to process
    :return: list of tokens
    """
    # return list of tokens
    return nltk.word_tokenize(str(text).lower())


def remove_stop_punct(tokenized_text):
    """
    :param tokenized_text: list of tokens
    :return: list of tokens without punctuation and stopwords
    """

    # initialize stopwords, punctuation and list for output
    stop_words = stopwords.words('english')
    punct = list(string.punctuation)
    filtered_sentence = []

    # if the input is empty, just return an empty list
    if isinstance(tokenized_text, list):
        pass
    else:
        return []

    # perform filtering
    for w in tokenized_text:
        if (w not in stop_words) and (w not in punct):
            filtered_sentence.append(w)

    # return the object
    return filtered_sentence


def omit(text, n):
    """
    :param text: string to be processed
    :param n: if text contains fewer characters than n (or if it is a default string),
    it is replaced with 'omitted'
    :return: input text or 'omitted'
    """

    # create list with default strings that imply omitting the MD&A
    omitted = ['Item 7. Management’s Discussion and Analysis of Financial Condition and Results of '
               'Operations.\nNot Applicable',
               'ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS',
               'Item 7. Management’s Discussion and Analysis of Financial Condition and Results of '
               'Operation\nOmitted.',
               'Item 7:\nManagement’s Discussion and Analysis of Financial Condition and Results of Operations',
               'Item 7. Management’s Discussion and Analysis of Financial Condition and Results of '
               'Operations.\nOmitted pursuant to General Instruction J of Form 10-K.',
               "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF "
               "OPERATIONS\nBecause of the limited business activity of the Company, the presentation of "
               "Management's Discussion and Analysis of Financial Condition and Results of Operations, "
               "as otherwise required by Item 303 of Regulation S-K, would not be meaningful. All relevant "
               "information is contained in the Monthly Reports (filed under Current Reports on Form 8-K) as "
               "described above.", "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF "
                                   "OPERATIONS\nNot Applicable"]

    # replace text with omitted if it is default string or when it has fewer than n characters
    if text in omitted:
        return 'omitted'
    if (len(str(text)) < n) & (text != 'missing'):
        return 'omitted'
    return text


def remove_low_frequency(tokenized_text, high_freq):
    """
    :param tokenized_text: list of tokens
    :param high_freq: list of high frequency words
    :return: filtered tokenized_text, only retain high freq words
    """

    # only retain high freq words
    filtered_text = list()
    for word in tokenized_text:
        if word in high_freq:
            filtered_text.append(word)
        else:
            filtered_text.append('_UNK_')
    # return filtered text
    return filtered_text


# LIBRARY - BATCH-LEVEL FUNCTIONS
# ______________________________________________________________________________________________________________________
def tokenize_batch(dataset, lemmatize):
    """
    :param dataset: dataframe to be processed in batch
    :param lemmatize: boolean, if true, tokenize and lemmatize, if false, only tokenize
    :return: dataframe with processed documents
    """

    # process documents
    if lemmatize:
        dataset['item_7'] = dataset['item_7'].apply(tokenize_lemmatize)
    else:
        dataset['item_7'] = dataset['item_7'].apply(tokenize)

    # return result
    return dataset


def remove_stop_punct_batch(dataset):
    """
    :param dataset: dataset to be processed in batch
    :return: dataset without stopwords and punctuation
    """
    # stopword removal
    dataset['item_7'] = dataset['item_7'].apply(remove_stop_punct)
    # return result
    return dataset


def omit_batch(dataset):
    """
    :param dataset: dataset to be processed in batch
    :return: dataset without ommited samples as 'omitted'
    """
    # omit
    dataset['item_7'] = dataset.apply(lambda x: omit(x['item_7'], 100), axis=1)
    # return result
    return dataset


def remove_low_frequency_batch(dataset, high_freq):
    """
    :param high_freq: high-frequency words to retain
    :param dataset: dataset to be processed in batch
    :return: dataset without low frequency tokens replaced with _unk_
    """
    # process
    dataset['item_7'] = dataset.apply(lambda x: remove_low_frequency(tokenized_text=x['item_7'],
                                                                     high_freq=high_freq), axis=1)
    # return result
    return dataset


def get_train_corpus(healthy, failed, last_data_year):
    """
    :param healthy: dataframe with 10-k reports from all companies not in BRD case table
    :param failed: dataframe with 10-k reports from all companies in BRD case table
    :param last_data_year: last year of data to be used during training
    :return: time-split training data
    """
    # split based on time
    # note that we adjust the labels in a later stage (during sampling - not needed here)
    htrain = healthy[healthy['year'] <= last_data_year]
    ftrain = failed[failed['year'] <= last_data_year]
    train = pd.concat([htrain, ftrain])
    train = train.reset_index(drop=True)

    # return the result
    return train
