# IMPORTS
# ______________________________________________________________________________________________________________________
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import nltk


# LIBRARY
# ______________________________________________________________________________________________________________________
def time_split_train(healthy, failed, last_data_year):
    """
    :param healthy: healthy df processed
    :param failed: failed df processed
    :param last_data_year: last year of data to be used during training (filing year)
    :return: train (a concat between healthy and failed but split based on time)
    """
    # split based on time
    htrain = healthy[healthy['year_filing'] <= last_data_year]
    ftrain = failed[failed['year_filing'] <= last_data_year]
    train = pd.concat([htrain, ftrain])
    train = train.reset_index(drop=True)

    # return the result
    return train


def time_split_holdout(healthy, failed, failure_years, n, first_year_holdout):
    """
    :param n: Fixed-length time window for prediction
    :param first_year_holdout: first year (of filing) for the holdout set
    :param healthy: healthy df processed
    :param failed: failed df processed
    :param failure_years: df with all the exact dates of failure
    :return: hol, a df containing the holdout set, distinguishable based on 'holdout_year' variable,
    for 2 consecutive years (of filing) (so 2 holdout sets are created in total)
    """

    # split on time
    hhol = healthy[healthy['year_filing'] <= first_year_holdout]
    fhol = failed[failed['year_filing'] <= first_year_holdout]
    # only use last n years
    hhol = hhol[hhol['year_filing'] >= (first_year_holdout - n + 1)]
    fhol = fhol[fhol['year_filing'] >= (first_year_holdout - n + 1)]

    # create full dataset
    hol19 = pd.concat([hhol, fhol])
    hol19 = hol19.reset_index(drop=True)

    # the companies that failed before the test period, remove
    failed_before_test = set(failure_years[failure_years['year_of_failure'] < first_year_holdout]['cik'])
    hol19 = hol19.drop(hol19[hol19['cik'].isin(failed_before_test)].index)

    # add in a column to distinguish the holdout years
    hol19['holdout_year'] = first_year_holdout

    # repeat for the second set
    hhol = healthy[healthy['year_filing'] <= first_year_holdout + 1]
    fhol = failed[failed['year_filing'] <= first_year_holdout + 1]

    # only use last n years
    hhol = hhol[hhol['year_filing'] >= (first_year_holdout + 1 - n + 1)]
    fhol = fhol[fhol['year_filing'] >= (first_year_holdout + 1 - n + 1)]

    # create full dataset
    hol20 = pd.concat([hhol, fhol])
    hol20 = hol20.reset_index(drop=True)

    # the companies that failed before the test period, remove
    failed_before_test = set(failure_years[failure_years['year_of_failure'] < first_year_holdout + 1]['cik'])
    hol20 = hol20.drop(hol20[hol20['cik'].isin(failed_before_test)].index)

    # add in a column to distinguish the holdout years
    hol20['holdout_year'] = first_year_holdout + 1

    # return the result
    return hol19, hol20


# ______________________________________________________________________________________________________________________


def impute_holdout(df, year, n):
    """
    :param df: Sub-dataframe that contains the data for a single company
                (E.g. when looping over all companies in the dataset and slicing on 'cik')
    :param year: year of the holdout set
    :param n: Fixed-length time window for prediction
    :return: Dataframe for single companies where the missing years are correctly imputed.
    """

    # retain cik of the company
    cik = df['cik'].iloc[0]

    # store the missing years in a list in three steps:

    # compute the years that should be present
    should_be_present = pd.Series(range(year - n + 1, year + 1, 1))
    # get all the years that are present
    are_present = list(df['year_filing'].unique())
    # find the missing ones
    missing = list(should_be_present[~should_be_present.isin(are_present)])

    # check if we need imputation
    if len(missing) > 0:
        # initialize a list to store the rows for the missing years
        rows = []

        # check if the company, for which we impute, failed
        if pd.notnull(df['failure_date'].iloc[0]):

            # retain date of failure
            try:
                failure_date = datetime.strptime(df['failure_date'].iloc[0], '%Y-%m-%d')
            except:
                failure_date = df['failure_date'].iloc[0].to_pydatetime()

            # perform imputation for each missing year
            for m_year in missing:

                # compute the date on which there should have been a filing
                try:
                    filing_date = df['filing_date'].iloc[0].replace(year=m_year)
                except:
                    filing_date = df['filing_date'].iloc[0].replace(year=m_year, day=1)
                filing_date = filing_date.to_pydatetime()

                # we adjust for binary labels in the 'sample_holdout' function
                # this labels assignment allows for the possibility of multiclass classification
                # this was not covered in the paper
                label = relativedelta(failure_date, filing_date).years + 1
                # remove imputations that are not allowed, e.g. after a bankruptcy
                # the actual removal is again performed in the 'sample_holdout' function
                if failure_date < filing_date:
                    label = -1

                # impute the row - mind the order of the variables
                row = [cik, 'missing', 'missing', filing_date, label, filing_date.year, 'missing',
                       df['failure_date'].iloc[0], year]
                rows.append(row)


        # healthy companies
        else:

            # perform imputation for each missing year
            for m_year in missing:

                # compute the date on which there should have been a filing
                try:
                    filing_date = df['filing_date'].iloc[0].replace(year=m_year)
                except:
                    filing_date = df['filing_date'].iloc[0].replace(year=m_year, day=1)

                # impute the row - mind the order of the variables
                row = [cik, 'missing', 'missing', filing_date, 0, filing_date.year, 'missing', np.datetime64('NaT'),
                       year]
                rows.append(row)

        # cast to dataframe, set columns and add new rows to original dataframe
        rows = pd.DataFrame(rows)
        rows.columns = df.columns
        df = df.append(rows)
        df = df.sort_values(['cik', 'year_filing']).reset_index(drop=True)

    return df


def sample_holdout(df, n, binary):
    """
    :param df:      Sub-dataframe that contains the data for a single company
                    (E.g. when looping over all companies in the dataset and slicing on 'cik')
    :param n:       Fixed-length time window for prediction
    :param binary:  Boolean to indicate if we want binary or multiclass labels - multiclass not covered in paper
    :return:        data, a list containing the n-sized firm-year samples used for prediction
    """

    # initialize a list to store the data in with the company cik
    data = [df.iloc[0]['cik']]

    # sort the dataframe according to the year in descending order
    df = df.sort_values(['cik', 'year_filing'], ascending=False).reset_index(drop=True)

    # loop over n
    for i in range(n):
        # retain the text in the correct order (df is sorted descending so most recent years are first)
        y = df.iloc[i]['item_7']
        # store the text in the list
        data.append(y)

    # add the label and mind the type
    label = df.iloc[0]['label']
    if type(label) == tuple:
        label = label[0]

    # additional check - normally these cases have been filtered out before when assigning the label
    # corresponds to bankrupt firms who restart activity after bankruptcy filing (or are currently in the process)

    # this was required since we wanted to allow for multiclass labels in the imputation function
    # can be simplified when only dealing with binary labels - not covered in paper
    if label < 0:
        return None

    # add label
    # this was required since we wanted to allow for multiclass labels in the imputation function
    # can be simplified when only dealing with binary labels - not covered in paper
    if binary:
        if label == 1:
            data.append(label)
        else:
            data.append(0)
    # if we use multiclass labels, add all
    else:
        data.append(label)

    return data


def dataset_transformer_holdout(dataset, n, binary, year):
    """
    :param dataset: The dataset to be transformed in the correct format of firm-year samples.
    :param n: Fixed-length time window for prediction
    :param year:    The year of the holdout set. This is needed for the imputation of missing years.
    :param binary:  Boolean to indicate if we want binary or multiclass labels - multiclass not covered in paper
    :return: dataset with all n-sized time-windows from the imputed input dataset.
    """

    # initialize a list to store the n-sized firm-year samples
    data = []

    # loop over each company in dataset
    cik_set = set(dataset['cik'].unique())
    for i, cik in enumerate(cik_set):

        # keep track of the progress
        if i % 50 == 0:
            print('Processing: [' + str(i) + ' / ' + str(len(cik_set)) + ']')

        # retain data only this company
        df = dataset[dataset['cik'] == cik]

        # impute the missing years
        df = impute_holdout(df, year, n)

        # sample from the holdout set

        # additional check - normally these cases have been filtered out before when assigning the label
        # corresponds to bankrupt firms who restart activity after bankruptcy filing
        # or are currently in the bankruptcy process
        if sample_holdout(df=df, n=n, binary=binary) is None:
            pass
        else:
            # add firm-year samples
            data.append(sample_holdout(df=df, n=n, binary=binary))

    # cast to list to pandas dataframe
    data = pd.DataFrame(data).reset_index(drop=True)

    # return result
    return data


# ______________________________________________________________________________________________________________________


def impute_train(df, last_data_year):
    """
    :param df:  Sub-dataframe that contains the data for a single company
                (E.g. when looping over all companies in the dataset and slicing on 'cik')
    :param last_data_year: last year used during training
    :return:    Dataframe for single companies where the missing years are correctly imputed.
    """

    # retain the cik of the company
    cik = df.iloc[0]['cik']
    # cast the year to an int
    df = df.astype({'year_filing': int})
    df = df.reset_index(drop=True)

    # compute the last date on which the company should have filed a 10k

    # check if the company failed
    if pd.notnull(df['failure_date'].iloc[0]):

        # compute how many years the company didn't file while they had to
        failure_date = df['failure_date'].iloc[0]
        filing_date = df.iloc[-1]['filing_date'] # the last filing date
        time_to_failure = relativedelta(failure_date, filing_date)
        years_not_filed = time_to_failure.years
        if years_not_filed > 0:
            last_date = filing_date.replace(year=filing_date.year + years_not_filed, day=1)
        else:
            last_date = filing_date
        # retain the year of the last filing
        max_year = last_date.year

    # healthy companies - should have filed always in the training period (assumption! see paper for motivation)
    else:
        max_year = last_data_year

    # compute the first date on which the company filed a 10k
    min_year = df['year_filing'].min()

    # compute the missing years in three steps

    # compute the years that should be present
    should_be_present = pd.Series(range(min_year, max_year + 1, 1))
    # get all the years that are present
    are_present = list(df['year_filing'].unique())
    # find the missing ones
    missing = list(should_be_present[~should_be_present.isin(are_present)])

    # check if we need imputation
    if len(missing) > 0:
        # initialize list to store rows for the missing years
        rows = []

        # check if the company failed
        if pd.notnull(df['failure_date'].iloc[0]):

            # get the date of failure
            try:
                failure_date = datetime.strptime(df['failure_date'].iloc[0], '%Y-%m-%d')
            except:
                failure_date = df['failure_date'].iloc[0].to_pydatetime()

            # perform imputation for each missing year
            for m_year in missing:

                # compute the date on which there should have been a filing
                filing_date = df['filing_date'].iloc[-1].replace(year=m_year, day=1)
                filing_date = filing_date.to_pydatetime()

                # we adjust for binary labels in the 'sliding window' function
                # this labels assignment allows for the possibility of multiclass classification
                # this was not covered in the paper
                label = relativedelta(failure_date, filing_date).years + 1
                # remove imputations that are not allowed, e.g. after a bankruptcy
                # the actual removal is again performed in the 'sliding window' function
                if failure_date < filing_date:
                    label = -1

                # impute the missing row, mind the order of the variables
                row = [cik, 'missing', 'missing', filing_date, label, filing_date.year, 'missing',
                       df['failure_date'].iloc[0]]
                rows.append(row)


        # healthy companies
        else:

            # perform imputation for each missing year
            for m_year in missing:

                # compute the date on which there should have been a filing
                filing_date = df['filing_date'].iloc[-1].replace(year=m_year, day=1)
                filing_date = filing_date.to_pydatetime()

                # impute the missing row, mind the order of the variables
                row = [cik, 'missing', 'missing', filing_date, 0, filing_date.year, 'missing', np.datetime64('NaT')]
                rows.append(row)

        # cast to dataframe, set columns and add new rows to original dataframe
        rows = pd.DataFrame(rows)
        rows.columns = df.columns
        df = df.append(rows)
        df = df.sort_values(['cik', 'year_filing']).reset_index(drop=True)

    return df


def did_not_exist_yet(df, n):
    """
    :param df:  Sub-dataframe that contains the data for a single company
                (E.g. when looping over all companies in the dataset and slicing on 'cik')
    :param n: Fixed-length time window for prediction
    :return: df: If a company, even after imputing the missing years, does not have n rows, they will be neglected.
    This causes a lot of failed firms that have existed (or were public) for only 1 or 2 years to be discarded.
    Therefore, we pad these dataframes with 'missing' rows the years before until they have len == n.
    This is representative for the testing/holdout scenario.
    """

    # sort the data according to filing year in ascending order (more recent years last)
    df = df.sort_values(['year_filing'], ascending=True).reset_index(drop=True)
    # first we compute the number of rows we need to add to reach length n
    to_pad = n - len(df)
    # now we extract the information from the first row that was available
    cik = df.iloc[0]['cik']
    first_year = df.iloc[0]['year_filing']
    first_date = df.iloc[0]['filing_date']

    # initialize a list to store the new rows in
    rows = []

    # loop over each row to add
    for i in range(to_pad):

        # check if the company failed
        if pd.notnull(df['failure_date'].iloc[0]):

            # get the date of failure
            try:
                failure_date = datetime.strptime(df['failure_date'].iloc[0], '%Y-%m-%d')
            except:
                failure_date = df['failure_date'].iloc[0].to_pydatetime()

            # compute the date on which they would have filed (had they existed)
            filing_date = first_date.replace(year=first_year - i - 1, day = 1)
            filing_date = filing_date.to_pydatetime()

            # we adjust for binary labels in the 'sliding window' function
            # this labels assignment allows for the possibility of multiclass classification
            # this was not covered in the paper
            label = relativedelta(failure_date, filing_date).years + 1

            # impute the missing row, mind the order of the variables
            row = [cik, 'missing', 'missing', label, filing_date, filing_date.year, 'missing', failure_date]
            rows.append(row)

        # healthy companies
        else:

            # compute the date on which they would have filed (had they existed)
            filing_date = first_date.replace(year=first_year - i - 1, day = 1)
            filing_date = filing_date.to_pydatetime()

            # impute the missing row, mind the order of the variables
            row = [cik, 'missing', 'missing', 0, filing_date, filing_date.year, 'missing', np.datetime64('NaT')]
            rows.append(row)

    # cast to dataframe, set columns and add new rows to original dataframe
    rows = pd.DataFrame(rows)
    rows.columns = df.columns
    df = df.append(rows)
    df = df.sort_values(['year_filing'])

    return df


def sliding_window(df, n, binary):
    """
    :param df:  Sub-dataframe that contains the data for a single company
                (E.g. when looping over all companies in the dataset and slicing on 'cik')
    :param n:   Fixed-length time window for prediction
    :param binary:  Boolean to indicate if we want binary or multiclass labels - multiclass not covered in paper
    :return:    data, a list of lists, where each 'inner' list represents an n-sized firm-year sample
    """
    # check if we need to add rows before the company existed / went public to reach size n
    if len(df) < n:
        df = did_not_exist_yet(df, n=n)

    # store the indices that can be used to extract n-sized firm-year samples
    first_start_index = n - 1
    last_start_index = len(df)

    # initialize a list to store the firm-year samples
    data = []

    # loop over all indices that make up valid firm-year samples
    for i in range(first_start_index, last_start_index, 1):
        # get the start and end index of the firm-year sample
        start = i - n + 1
        end = i + 1

        # setup list to hold the data of a single firm-year sample and store cik
        window = [df.iloc[0]['cik']]

        # get the firm-year samples data in a dataframe
        sample = df[start:end]
        # invert the order to add more recent data first
        sample = sample.iloc[::-1]
        # retain the text of the firm-year sample and add it to the list (window)
        counter = 0
        for i, row in sample.iterrows():
            # extract the text from the row
            y = row['item_7']
            # append this to the firm-year sample list
            window.append(y)

            # get the label from the most recent observation in the sliding window (year_1)
            if counter == 0:
                label = row['label']
                counter += 1

        # add label
        # this was required since we wanted to allow for multiclass labels in the imputation function
        # can be simplified when only dealing with binary labels - not covered in paper
        if binary:
            if label == 1:
                window.append(label)
            else:
                window.append(0)
        # if we use multiclass labels, add all
        else:
            window.append(label)

        # add firm-year sample to data
        data.append(window)

    # return the result
    return data


def dataset_transformer_train(dataset, n, binary, last_data_year):
    """
    :param last_data_year: last year of data used during training
    :param dataset: The dataset to be transformed in the correct format.
    :param n: Fixed-length time window for prediction
    :param binary:  Boolean to indicate if we want binary or multiclass labels - multiclass not covered in paper
    :return: dataset with all n-sized firm-year samples from the imputed input dataset.
    """

    # initialize a list to store the n-sized firm-year samples in
    data = []

    # we loop over each company in the dataset
    cik_set = set(dataset['cik'].unique())
    for i, cik in enumerate(cik_set):

        # track progress
        if i % 50 == 0:
            print('Processing: [' + str(i) + ' / ' + str(len(cik_set)) + ']')

        # retain the data for this firm
        df = dataset[dataset['cik'] == cik]

        # impute missing rows
        df = impute_train(df, last_data_year)

        # add firm-year samples for this company
        data.extend(sliding_window(df, n, binary))

    # transform the result in a pandas dataframe
    data = pd.DataFrame(data)

    # return the result
    return data


# ______________________________________________________________________________________________________________________


def tokenize(text):
    """
    :param text: input string to process
    :return: list of tokens
    """
    # return list of tokens
    return nltk.word_tokenize(str(text).lower())


def remove_missing_rows(dataset, n, frac):
    """
    :param dataset: a pandas dataframe with rows where all years are 'missing'
    :param frac: fraction of missing rows to be removed
    :param n: fixed-length time window
    :return: dataframe with frac of  rows only containing 'missing' removed
    """
    # First we reset the index since we will drop on index
    dataset = dataset.reset_index(drop=True)

    # remove the rows with all missing values
    if n == 3:
        idx = dataset[(dataset['year_1'] == 'missing') & (dataset['year_2'] == 'missing') & (
                dataset['year_3'] == 'missing')].sample(frac=frac).index
        dataset = dataset.drop(idx)
        dataset = dataset.reset_index(drop=True)
    if n == 1:
        idx = dataset[(dataset['year_1'] == 'missing')].sample(frac=frac).index
        dataset = dataset.drop(idx)
        dataset = dataset.reset_index(drop=True)

    # return the output
    return dataset
