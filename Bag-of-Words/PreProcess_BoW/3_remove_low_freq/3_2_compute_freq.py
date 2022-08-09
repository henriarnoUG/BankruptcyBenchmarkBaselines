# IMPORTS
# ______________________________________________________________________________________________________________________
from ..preprocessing_library import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''
# number of words to retain in vocab
most_common = 50000

# SCRIPT
# ______________________________________________________________________________________________________________________
print('READING IN THE DATA')

# specify location
data_location = data_location + 'intermediate_processed/'
# read data
data = pd.read_csv(data_location + 'frequencies/train.csv', index_col=0, chunksize=10)

# initialize a frequency object
frequencies = nltk.FreqDist()

# in batch, count the word occurrences and add this to the frequencies
for i, df in enumerate(data):
    # track progress
    if i % 100 == 0:
        print('processing chunk ' + str(i))

    # store the vocab of the current chunk in a list
    vocab = []
    for text in df['item_7'].apply(lambda x: str(x).split(' ')):
        vocab.extend(text)

    # compute frequencies and append to full frequencies
    freq = nltk.FreqDist(vocab)
    frequencies.update(freq)

    # save intermediate freqs
    if i % 1000 == 0:
        with open(data_location + 'frequencies/frequencies', "wb") as fp:  # Pickling
            pickle.dump(frequencies, fp)

# compute most_common most common words
top_freq = frequencies.most_common(most_common)
# cast to list
words = []
for (word, freq) in top_freq:
    words.append(word)

# store final output
with open(data_location + 'frequencies/frequencies', "wb") as fp:  # Pickling
    pickle.dump(frequencies, fp)

with open(data_location + 'frequencies/most_common_list', "wb") as fp:  # Pickling
    pickle.dump(words, fp)
