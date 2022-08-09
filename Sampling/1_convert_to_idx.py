# IMPORTS
# ______________________________________________________________________________________________________________________
import os
import pandas as pd
import time

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''
model = 'bow' # 'bow' or 'transformers'

# SCRIPT
# ______________________________________________________________________________________________________________________
# specify location
data_location = data_location + 'intermediate_processed/'

# read in the data
print('READING IN THE DATA')
healthy = pd.read_csv(data_location + 'healthy_' + model + '.csv', index_col = 0)
failed = pd.read_csv(data_location + 'failed_' + model + '.csv', index_col = 0).reset_index(drop=True)

# sort values
healthy = healthy.sort_values(['cik', 'year']).reset_index(drop=True)
failed = failed.sort_values(['cik', 'year']).reset_index(drop=True)

# add index as document identifier
start_idx_failed = len(healthy)
healthy['doc_id'] = healthy.index
failed['doc_id'] = failed.index + start_idx_failed

# store
healthy.to_csv(data_location + 'healthy_' + model + '.csv')
failed.to_csv(data_location + 'failed_' + model + '.csv')