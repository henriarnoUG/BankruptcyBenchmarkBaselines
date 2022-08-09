# IMPORTS
# ______________________________________________________________________________________________________________________
from sampling_library import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''

# model to use
model = 'bow' # 'bow' or 'transformers'

# history of documents used to make a prediction
# (1 or 3) in the paper
n = 3

# development set or test set
dev = False

# time period to use
dev_train = 2015 # last year (of filing) used to train (in development)
dev_hol = 2017 # first year (of filing) used for holdout set (in development)
test_train = 2017 # last year (of filing) used to train (in test)
test_hol = 2019 # first year (of filing) used for holdout set (in test)

# fraction of 'missing' rows to delete from train, NOT from holdout
# see relevant section in paper for motivation
frac = 0.95

# SCRIPT
# ______________________________________________________________________________________________________________________

# specify store location
store_location = data_location + 'model_data/' + model + '/'
try:
    os.mkdir(store_location)
except:
    pass

# read in the data
print('READING IN THE PROCESSED DATA - ALREADY CONTAINS DOC-ID')
healthy = pd.read_csv(data_location + 'intermediate_processed/healthy_' + model + '.csv', index_col = 0)\
    .reset_index(drop=True)
failed = pd.read_csv(data_location + 'intermediate_processed/failed_' + model + '.csv', index_col = 0)\
    .reset_index(drop=True)
failure_years = pd.read_csv(data_location + 'failure_years.csv', index_col=0).reset_index(drop=True)


# only use doc_id to sample and delete text
# when using a long history this is more memory efficient (otherwise a lot of redundancy)
healthy['item_7'] = healthy['doc_id']
failed['item_7'] = failed['doc_id']
healthy = healthy.drop(['doc_id'], axis = 1)
failed = failed.drop(['doc_id'], axis = 1)

# cast all dates to datetime
healthy['filing_date'] = pd.to_datetime(healthy['filing_date'])
healthy['period_of_report'] = pd.to_datetime(healthy['period_of_report'])
failed['filing_date'] = pd.to_datetime(failed['filing_date'])
failed['period_of_report'] = pd.to_datetime(failed['period_of_report'])
failed['failure_date'] = pd.to_datetime(failed['failure_date'])

# change year variable in filing / reporting year
healthy['year_filing'] = healthy['filing_date'].apply(lambda x : x.year)
failed['year_filing'] = failed['filing_date'].apply(lambda x : x.year)
healthy['year_report'] = healthy['period_of_report'].apply(lambda x : x.year)
failed['year_report'] = failed['period_of_report'].apply(lambda x : x.year)
failed = failed.drop('year', axis = 1)
healthy = healthy.drop('year', axis = 1)

# add a None value for the failure date of healthy companies
healthy['failure_date'] = None

# ______________________________________________________________________________________________________________________

print('PERFORMING TIME-SPLIT')
# split based on time, first we set the dates as specified by the user
if dev:
    last_data_year_train = dev_train
    first_year_holdout = dev_hol
else:
    last_data_year_train = test_train
    first_year_holdout = test_hol

# create the training data
train = time_split_train(healthy, failed, last_data_year_train)

# create the holdout data
holdout19, holdout20 = time_split_holdout(healthy, failed, failure_years, n=5,
                                          first_year_holdout=first_year_holdout)

# ______________________________________________________________________________________________________________________

print('SAMPLING HOLDOUT')
print('SET 1:')
holdout19_model = dataset_transformer_holdout(dataset=holdout19, n=5, binary=True,
                                              year=first_year_holdout)
print('SET 2:')
holdout20_model = dataset_transformer_holdout(dataset=holdout20, n=5, binary=True,
                                              year=first_year_holdout+1)

# adjust names
holdout19_model.columns = ['cik', 'year_1', 'year_2', 'year_3', 'year_4', 'year_5', 'label']
holdout20_model.columns = ['cik', 'year_1', 'year_2', 'year_3', 'year_4', 'year_5', 'label']

# select relevant history
if n == 3:
    holdout19_model = holdout19_model[['cik', 'year_1', 'year_2', 'year_3', 'label']]
    holdout20_model = holdout20_model[['cik', 'year_1', 'year_2', 'year_3', 'label']]
else:
    holdout19_model = holdout19_model[['cik', 'year_1', 'label']]
    holdout20_model = holdout20_model[['cik', 'year_1', 'label']]

# ______________________________________________________________________________________________________________________

print('SAMPLING TRAIN')
train_model = dataset_transformer_train(train, n, True, last_data_year=last_data_year_train)

# adjust column names
if n == 3:
    train_model.columns = ['cik', 'year_1', 'year_2', 'year_3', 'label']
else:
    train_model.columns = ['cik', 'year_1', 'label']

# ______________________________________________________________________________________________________________________

print('DROPPING ALL MISSING ROWS')
# drop missing rows where all text is missing
train_model = remove_missing_rows(train_model, n, frac=frac)

# ______________________________________________________________________________________________________________________

print('STORING EVERYTHING')
# add in a year to keep the different holdout sets separate
holdout19_model['holdout_year'] = first_year_holdout
holdout20_model['holdout_year'] = first_year_holdout+1
holdout = pd.concat([holdout19_model, holdout20_model])

# reset the index
holdout = holdout.reset_index(drop=True)
train_model = train_model.reset_index(drop = True)


if dev:
    holdout.to_csv(store_location + 'dev_' + model + '_' + str(n) + '.csv')
    train_model.to_csv(store_location + 'train_dev_' + model + '_' + str(n) + '.csv')
else:
    holdout.to_csv(store_location + 'holdout_' + model + '_' + str(n) + '.csv')
    train_model.to_csv(store_location + 'train_full_' + model + '_' + str(n) + '.csv')
