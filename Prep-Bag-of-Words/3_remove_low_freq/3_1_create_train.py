# IMPORTS
# ______________________________________________________________________________________________________________________
from ..preprocessing_library import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''
# last year to be used in train set
last_data_year_train = 2017


# SCRIPT
# ______________________________________________________________________________________________________________________
# create location to store intermediate files
os.mkdir(data_location + 'intermediate_processed/frequencies/')

print('READING IN THE DATA')
# read data
healthy = pd.read_csv(data_location + 'intermediate_processed/healthy_processed_complete.csv', index_col=0)
failed = pd.read_csv(data_location + 'intermediate_processed/failed_processed_complete.csv', index_col=0)
# create and store train data used to compute frequencies
train = get_train_corpus(healthy, failed, last_data_year_train)
train.to_csv(data_location + 'intermediate_processed/frequencies/train.csv')
