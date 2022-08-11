# IMPORTS
# ______________________________________________________________________________________________________________________
from preprocessing_library import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ""

# SCRIPT
# ______________________________________________________________________________________________________________________
# create location to store intermediate files
os.mkdir(data_location + 'intermediate_processed/')
os.mkdir(data_location + 'intermediate_processed/chunks/')


print('PROCESSING FAILED DATA')
# read in all filings of companies in BRD case table
failed = pd.read_csv(data_location + 'failed_text_all.csv', index_col=0)
# omit
failed = omit_batch(failed)
# tokenize lemmatize
failed = tokenize_batch(failed, True)
# stopwords and punct
failed = remove_stop_punct_batch(failed)
# untokenize for storage
failed['item_7'] = failed['item_7'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
# store
failed.to_csv(data_location + "intermediate_processed/failed_processed.csv")

# free up memory
del failed
gc.collect()
# ______________________________________________________________________________________________________________________

print('PROCESSING HEALTHY DATA')
# read in filings of companies not in BRD case table in chunks of 5000 for efficient memory usage
data = pd.read_csv(data_location + 'healthy_text_all.csv', index_col=0, chunksize=5000)

for i, healthy in enumerate(data):
    print('PROCESSING CHUNK: ' + str(i))
    # omit
    healthy = omit_batch(healthy)
    # tokenize lemmatize
    healthy = tokenize_batch(healthy, True)
    # stopwords and punct
    healthy = remove_stop_punct_batch(healthy)
    # untokenize for storage
    healthy['item_7'] = healthy['item_7'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    # store
    healthy.to_csv(data_location + 'intermediate_processed/chunks/chunk_' + str(i) + ".csv")

# free up memory
del healthy, data
gc.collect()
# ______________________________________________________________________________________________________________________

print('PATCHING HEALTHY DATA')
# init list to store data
data = []

# loop over each chunk
for file in os.listdir(data_location + 'intermediate_processed/chunks/'):
    print(file)
    # read in each chunk
    filepath = data_location + 'intermediate_processed/chunks/' + file
    df = pd.read_csv(filepath, index_col=0)
    df = df.reset_index(drop=True)
    # append to data
    data.append(df)

# create complete pandas dataframe
healthy = pd.concat(data)
healthy = healthy.reset_index(drop = True)
# store
healthy.to_csv(data_location + "intermediate_processed/healthy_processed.csv")
