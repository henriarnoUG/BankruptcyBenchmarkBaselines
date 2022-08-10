# IMPORTS
# ______________________________________________________________________________________________________________________
from encoding_library import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''

# SCRIPT
# ______________________________________________________________________________________________________________________

# specify location
path = data_location + 'intermediate_processed/'
path_w2v = data_location + 'model_data/W2V_data/'


print('READING THE DATA - PROCESSED DOCUMENTS AND W2V MODEL')
# read w2v and add _PAD_ token as all zero vector
w2v = Word2Vec.load(path_w2v + 'w2v.model')
w2v.wv.add_vector('_PAD_', np.zeros(100))
# read in the documents as iterator objects
healthy = pd.read_csv(path + 'healthy_bow.csv', index_col=0, chunksize=100, header=0)
failed = pd.read_csv(path + 'failed_bow.csv', index_col=0, chunksize=100, header=0)

# perform the encoding and save the output
print('ENCODING HEALTHY DOCS')
encode_word2vec(healthy, 'healthy', data_location, w2v)
print('ENCODING FAILED DOCS')
encode_word2vec(failed, 'failed', data_location, w2v)


