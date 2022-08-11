# IMPORTS
# ______________________________________________________________________________________________________________________
from ..preprocessing_library import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''

# SCRIPT
# ______________________________________________________________________________________________________________________
# location to store the results
data_location = data_location + 'intermediate_processed/'
store_location = data_location + 'frequencies/chunks/'
os.mkdir(store_location)


print('READING IN THE DATA')
# read in the data
with open(data_location + '/frequencies/most_common_list', "rb") as fp:
    most_common = pickle.load(fp)
data = pd.read_csv(data_location + 'healthy_processed_complete.csv', index_col=0, chunksize=5000)

for i, healthy in enumerate(data):
    # track progress
    print('PROCESSING CHUNK: ' + str(i))
    # tokenize
    healthy = tokenize_batch(healthy, False)
    # replace uncommon words
    healthy = remove_low_frequency_batch(healthy, most_common)
    # untokenize for storage
    healthy['item_7'] = healthy['item_7'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    # store
    healthy.to_csv(store_location + 'chunk_' + str(i) + ".csv")

print('PATCHING HEALTHY DATA')
# init list to store data
data = []

# loop over each chunk
for file in os.listdir(data_location + 'frequencies/chunks/'):
    # read chunk
    filepath = data_location + 'frequencies/chunks/' + file
    df = pd.read_csv(filepath, index_col=0).reset_index(drop=True)
    # append to data
    data.append(df)

# create full df
healthy = pd.concat(data)
# store
healthy.reset_index(drop=True).to_csv(data_location + "healthy_bow.csv")

print('PROCESSING FAILED DATA')
# read in the data
data = pd.read_csv(data_location + 'failed_processed_complete.csv', index_col=0, chunksize=5000)

for i, failed in enumerate(data):
    print('PROCESSING CHUNK: ' + str(i))
    # tokenize
    failed = tokenize_batch(failed, False)
    # replace uncommon words
    failed = remove_low_frequency_batch(failed, most_common)
    # untokenize for storage
    failed['item_7'] = failed['item_7'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    # store (should be only 1 chunk)
    if i == 0:
        failed.to_csv(data_location + 'failed_bow.csv')
    else:
        failed.to_csv(store_location + 'failed_chunk_' + str(i) + ".csv")