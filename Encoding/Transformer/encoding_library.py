# IMPORTS
# ______________________________________________________________________________________________________________________
import pandas as pd
import os
import torch
from transformers import LongformerTokenizer, LongformerModel, BigBirdModel
import warnings
from torch.utils.data import Dataset
import time
import numpy as np

# LIBRARY
# ______________________________________________________________________________________________________________________


def encode(seq, tknzr, transformer, device):
    """
    :param device: device to put tensors on (cuda or cpu)
    :param seq: the sequence to encode
    :param tknzr: the tokenizer from the Huggingface library
    :param transformer: the transformer model from the Huggingface library
    :return: The encoded sequence in a numpy array
    """
    # tokenize the input
    input_ids = torch.tensor(tknzr.encode(seq, truncation=True)).unsqueeze(0)  # batch of size 1
    # to device
    input_ids = input_ids.to(device)
    # pass through pretrained model
    outputs = transformer(input_ids)
    # convert to numpy
    tensor = outputs.pooler_output.squeeze(0)
    tensor_np = tensor.cpu().detach().numpy()
    # del memory intensive tensors
    del input_ids, tensor, outputs
    torch.cuda.empty_cache()

    # return the result
    return tensor_np


def encode_batch(iterator, tknzr, transformer, data_location, name, device):
    """
    :param device: device to put tensors on (cuda or cpu)
    :param data_location: location of the files created by the DSI code
    :param name: a name that is appended to the name of the output files
    :param iterator: an iterator that runs through the documents to encode
    :param tknzr: the tokenizer from the Huggingface library
    :param transformer: the transformer model from the Huggingface library
    :return: void function that stores two lists, the first containing the encoded tensors, the second
    containing the doc_id of the corresponding tensor.
    """

    # specify storage location
    path_store = data_location + 'model_data/transformers/'

    # init a list to store the encodings
    # init a list to store doc_id
    tensors = []
    doc_ids = []

    # loop over the documents in the iterator
    for i, df in enumerate(iterator):

        # keep track of the progress
        print('Processing chunk: ' + str(i) + ' with chunksize 100')
        start_time = time.time()

        # loop over each document in the chunk
        for j, row in df.iterrows():

            # perform the encoding
            encoding = encode(row['item_7'], tknzr, transformer, device)
            tensors.append(encoding)

            # store the doc id
            doc_ids.append(row['doc_id'])

        # store intermediate output
        if i % 50 == 0:
            print('--- [storing intermediate output] ---')
            # store
            np.save(path_store + 'tensors_' + name + '.npy', tensors, allow_pickle=True)
            np.save(path_store + 'doc_ids_' + name + '.npy', doc_ids, allow_pickle=True)

    # store final result
    print('--- [storing intermediate output] ---')
    # store
    np.save(path_store + 'tensors_' + name + '.npy', tensors, allow_pickle=True)
    np.save(path_store + 'doc_ids_' + name + '.npy', doc_ids, allow_pickle=True)

    # print duration
    print('Total duration:')
    print(time.time() - start_time)
