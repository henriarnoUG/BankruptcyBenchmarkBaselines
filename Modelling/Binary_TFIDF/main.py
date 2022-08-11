# IMPORTS
# ______________________________________________________________________________________________________________________
from library import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''
# name to create the results folder
name = 'binary'  # 'binary' or 'tfidf'
# development or test setting
dev = True
# number of documents used for prediction
n = 1 # 1 or 3
# resample the training data with balance_prop % of failed firm-year samples (e.g. 5%, 10% or 50%)
# uses under-sampling of the majority class
balance = True
balance_prop = 10

# hyperparameter ranges for (see sklearn documentation for more details) (https://scikit-learn.org/stable/)
# k: the number of feature to retain after uni-variate feature selection (select k-best)
# C: the inverse L2-regularisation strength
k_range = [100, 1000, 10000, 25000, 'all']
C_range = [1e-05, 1e-03, 0.05, 0.1, 0.15, 0.5, 1, 10, 100, 1000, 5000]

# the range of n_grams to use (e.g. (1,2) means uni- and bi-grams)
n_grams = (1, 2)
# the feature extraction method, CountVectorizer creates binary features, TfidfVectorizer creates TFIDF features.
feature = CountVectorizer(binary=True, ngram_range=n_grams)
# feature = TfidfVectorizer(ngram_range=n_grams)


# SCRIPT
# ______________________________________________________________________________________________________________________

# specify location to store the results
save_location = data_location + 'results_' + name + '/'
store_location = data_location + 'intermediate_processed/'
try:
    os.mkdir(save_location)
except:
    pass


print('READING IN THE DATA')
# read in the firm-year dataset
if dev:
    train = pd.read_csv(data_location + 'model_data/bow/train_dev_bow_' + str(n) + '.csv', index_col=0)
    holdout = pd.read_csv(data_location + 'model_data/bow/dev_bow_' + str(n) + '.csv', index_col=0)
    # split up the holdout set
    holdout1 = holdout[holdout['holdout_year'] == 2017]
    holdout2 = holdout[holdout['holdout_year'] == 2018]

else:
    train = pd.read_csv(data_location + 'model_data/bow/train_full_bow_' + str(n) + '.csv', index_col=0)
    holdout = pd.read_csv(data_location + 'model_data/bow/holdout_bow_' + str(n) + '.csv', index_col=0)
    # split up the holdout set
    holdout1 = holdout[holdout['holdout_year'] == 2019]
    holdout2 = holdout[holdout['holdout_year'] == 2020]

# read in the documents to process
healthy = pd.read_csv(store_location + 'healthy_bow.csv', index_col=0).reset_index(drop=True)
failed = pd.read_csv(store_location + 'failed_bow.csv', index_col=0).reset_index(drop=True)

# concatenate the healthy and failed documents into a single dataframe
# also add an entry for the missing documents, adjust the datatypes and select the relevant columns
all_docs = pd.concat([healthy, failed]).reset_index(drop=True)
all_docs = all_docs.astype({'doc_id': str})
all_docs = all_docs[['item_7', 'doc_id']]
all_docs = all_docs.append(pd.DataFrame({'item_7': 'missing', 'doc_id': 'missing'}, index=[len(all_docs)]))
del healthy, failed


# ______________________________________________________________________________________________________________________

print('RESAMPLING')
# balance out the class proportions if specified
if balance:
    train = balance_train(train, balance_prop)

print('SWAPPING DOC IDS FOR CORRESPONDING TEXT')

# as a first step, we indicate which columns in the firm-year dataset contain doc ids and need to be
# swapped with the corresponding text
if n == 1:
    variables_to_swap = ['year_1']
else:
    variables_to_swap = ['year_1', 'year_2', 'year_3']

# now, we perform the swap for each variable
for variable in variables_to_swap:
    # left merge the firm-year sample dataset with the document dataset on doc_id
    train = pd.merge(train, all_docs, left_on=variable, right_on='doc_id', how='left')
    holdout = pd.merge(holdout, all_docs, left_on=variable, right_on='doc_id', how='left')
    # rename the variable to the corresponding text
    train[variable] = train['item_7']
    holdout[variable] = holdout['item_7']
    # remove the unnecessary variables
    train = train.drop(['item_7', 'doc_id'], axis=1)
    holdout = holdout.drop(['item_7', 'doc_id'], axis=1)
    # cast to string
    train = train.astype({variable: str})
    holdout = holdout.astype({variable: str})

# transform the dataframes into numpy arrays before training
# simply concatenate documents when dealing with classification based on multiple documents

# single document for prediction
if n == 1:
    X_train = np.array(train['year_1'])
    y_train = np.array(train['label'])

    X_hol1 = np.array(holdout1['year_1'])
    y_hol1 = np.array(holdout1['label'])

    X_hol2 = np.array(holdout2['year_1'])
    y_hol2 = np.array(holdout2['label'])

    # also store the company IDs for analysis later on
    cik_1 = np.array(holdout1['cik'])
    cik_2 = np.array(holdout2['cik'])

# multiple documents for prediction
elif n == 3:
    X_train = np.array(train['year_1'] + ' ' + train['year_2'] + ' ' + train['year_3'])
    y_train = np.array(train['label'])

    X_hol1 = np.array(holdout1['year_1'] + ' ' + holdout1['year_2'] + ' ' + holdout1['year_3'])
    y_hol1 = np.array(holdout1['label'])

    X_hol2 = np.array(holdout2['year_1'] + ' ' + holdout2['year_2'] + ' ' + holdout2['year_3'])
    y_hol2 = np.array(holdout2['label'])

    # also store the company IDs for analysis later on
    cik_1 = np.array(holdout1['cik'])
    cik_2 = np.array(holdout2['cik'])

# ______________________________________________________________________________________________________________________

# For the modelling part, we loop over each hyperparameter setting, build a model and report results
for k in k_range:
    for C in C_range:

        # track progress
        print('SETTINGS: --- K --- ' + str(k) + ' --- C --- ' + str(C))

        # train the model
        model = build_model(X_train, y_train, feature_constructor=feature, C=C, k=k)

        # create a folder to store the results
        try:
            dir = save_location + str(k) + '_' + str(C) + '/'
            os.mkdir(dir)
        except:
            pass

        # make predictions and store the result
        preds_0, preds_1 = evaluate(X_hol1, model)
        preds = pd.DataFrame({'preds_0': preds_0, 'preds_1': preds_1, 'cik': cik_1})
        preds.to_csv(dir + 'predictions_holdout_set_1.csv')

        # repeat for the second holdout set
        preds_0, preds_2 = evaluate(X_hol2, model)
        preds = pd.DataFrame({'preds_0': preds_0, 'preds_1': preds_2, 'cik': cik_2})
        preds.to_csv(dir + 'predictions_holdout_set_2.csv')

        # create and store feature weights
        feature_imp = get_feature_weights(model)
        feature_imp.to_csv(dir + 'feature_imp.csv')

