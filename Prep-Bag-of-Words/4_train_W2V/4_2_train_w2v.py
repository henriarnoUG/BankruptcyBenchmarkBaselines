# IMPORTS
# ______________________________________________________________________________________________________________________
import os

from ..preprocessing_library import *


# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''

# W2V params
sg=1
window=5
hs=0
negative=5
vector_size=100
min_count = 1 # already replaced low-freq words

# SCRIPT
# ______________________________________________________________________________________________________________________

# specify location
store_location = data_location + 'model_data/'
os.mkdir(store_location)
store_location = store_location + 'W2V_data/'
os.mkdir(store_location)
data_location = data_location + 'intermediate_processed/'


# read in training data in chunks of 1000 for memory efficiency
train = pd.read_csv(data_location + 'train_w2v.csv', index_col=0, chunksize=1000)

# initialize w2v model and track progress
w2v = None
start_time = time.time()

print('TRAINING W2V - TAKES LONG')
for i, df in enumerate(train):
    print("PROCESSING CHUNK " + str(i))
    # retokenize data
    data = list(df['item_7'].apply(tokenize))

    # create initial w2v model
    if w2v is None:
        w2v = gensim.models.Word2Vec(sentences=data, sg=sg, window=window, hs=hs, negative=negative,
                                     vector_size=vector_size, min_count=min_count)
    # extend model with data from new chunks
    else:
        w2v.build_vocab(data, update=True)
        w2v.train(data, total_examples=len(data), epochs=w2v.epochs)

    # checkpoint model
    if i % 20 == 0:
        w2v.save(store_location + 'w2v.model')

# store final model
print('STORING W2V')
w2v.save(store_location + 'w2v.model')

# print total training duration
print(str(round(time.time() - start_time, 2) / 60), ' minutes')
