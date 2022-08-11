# IMPORTS
# ______________________________________________________________________________________________________________________
import os
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import torch
import nltk


# LIBRARY
# ______________________________________________________________________________________________________________________

def tokenize(text):
    """
    :param text: input string to tokenize
    :return: list of tokens
    """
    # return list of tokens
    return nltk.word_tokenize(str(text).lower())


def text_to_idx(text, model, pad_length):
    """
    :param text: the text to encode with the word2vec model
    :param model: a pretrained word2vec model from the gensim package
    :param pad_length: the length that is used to pad the documents,
    (!) when computing mean embeddings, we do not take padding into account. (!)
    [we  then set this parameter very high (e.g. 10.000)
    and discard zero vectors (i.e. the embeddings of the _PAD_ token).]
    :return: a tensor with (pad_length) as shape. Each token is replaced with its index in the word2vec model
    """

    # store the dictionary with the tokens as keys and the embeddings as values
    mapping = model.wv.key_to_index
    # tokenize
    text = tokenize(str(text))
    # compute the required number of padding tokens
    to_pad = pad_length - len(text)

    # documents that are too long
    if to_pad < 0:
        # only select first pad_length tokens
        text = text[:pad_length]

    # documents that are too short
    else:

        # add _PAD_ token until pad_length is reached
        counter = 0
        while counter < to_pad:
            text.append('_PAD_')
            counter += 1

    # map each token to its corresponding index in the w2v model
    result = [mapping.get(k, mapping['_unk_']) for k in text]
    # convert to tensor
    result = torch.IntTensor(result)

    return result


def mean_embeddings(tensor, input_size):
    """
    :param tensor: a tensor with (batch_size, 1, pad_length, representation_length) as shape
    :param input_size: representation_length (can be deduced from tensor.shape as well)
    :return: the mean embeddings of all the documents in the input tensor
    """

    # compute the number of tokens in each document that are not _PAD_ (original number of embedded tokens)
    nonzero = (tensor.count_nonzero(dim=3) == input_size).sum(2)
    # add singleton dimension for broadcasting
    nonzero = nonzero.unsqueeze(2)
    # sum the embeddings of each token in a document
    rep_sum = tensor.sum(2)
    # compute the mean embedding
    mean_emb = rep_sum / nonzero

    return mean_emb


def encode_word2vec(iterator, name, data_location, w2v):
    """
    :param w2v: a pretrained word2vec model from the gensim package
    :param data_location: location of the files created by the DSI code
    :param iterator: an iterator that runs over the documents in chunks
    :param name: a name that is appended to the name of the output files
    :return: void function that stores two lists, the first containing the mean embedding tensors, the second
    containing the doc_id of the corresponding tensor.
    """

    # specify location
    path_store = data_location + 'model_data/bow/'

    # init a PyTorch embedding layer and set the padding token and freeze the weights (strictly not required)
    weights = torch.FloatTensor(w2v.wv.vectors)
    pad_idx = w2v.wv.key_to_index['_PAD_']
    embedding_layer = torch.nn.Embedding.from_pretrained(weights, padding_idx=pad_idx)
    embedding_layer.weight.requires_grad = False

    # init a list to store the mean embedding tensors
    # init a list to store doc_id
    tensors = []
    doc_ids = []

    # loop over the documents in the iterator
    for i, df in enumerate(iterator):

        # track progress
        print('Processing chunk: ' + str(i) + ' with chunksize 100')

        # initialize a list to store the word2vec indices of the tokens in the documents in
        indices = []
        # loop over each document
        for j, row in df.iterrows():
            # compute the word2vec indices and add to the list
            indices.append(text_to_idx(row['item_7'], w2v, 10000))
            # add doc_id
            doc_ids.append(row['doc_id'])

        # compute the mean embeddings in several steps

        # cast to tensor
        indices = torch.cat(indices)
        indices = indices.reshape(len(df), 10000)
        # embed the indices and add singleton dimension (required by mean_embeddings function)
        embeddings = embedding_layer(indices)
        embeddings = embeddings.unsqueeze(1)
        # compute mean embeddings
        average_embeddings = mean_embeddings(embeddings, 100)
        # format the output
        average_embeddings = average_embeddings.squeeze(1)
        average_embeddings = average_embeddings.detach().numpy()
        # add tensors to list
        tensors.extend(average_embeddings)

        # store intermediate output
        if i % 50 == 0:
            print('--- [storing intermediate output] ---')
            # store
            np.save(path_store + 'tensors_' + name + '.npy', tensors, allow_pickle=True)
            np.save(path_store + 'doc_ids_' + name + '.npy', doc_ids, allow_pickle=True)

    # store final output
    np.save(path_store + 'tensors_' + name + '.npy', tensors, allow_pickle=True)
    np.save(path_store + 'doc_ids_' + name + '.npy', doc_ids, allow_pickle=True)
