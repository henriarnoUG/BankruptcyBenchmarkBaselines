# IMPORTS
# ______________________________________________________________________________________________________________________
import pandas as pd
from dateutil.relativedelta import relativedelta

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
path = ''

# SCRIPT
# ______________________________________________________________________________________________________________________
# read in data
failed = pd.read_csv(path + 'failed_text_all.csv', index_col=0, header=0).reset_index(drop=True)
healthy = pd.read_csv(path + 'healthy_text_all.csv', index_col=0).reset_index(drop=True)
failure_years = pd.read_csv(path + 'failure_years.csv', index_col=0).reset_index(drop=True)

# drop all nan row, drop decimals from cik
failed = failed.drop(0)
failed = failed.astype({'cik': int})
failed = failed.astype({'cik': str})
failure_years = failure_years.astype({'cik': int})
failure_years = failure_years.astype({'cik': str})
healthy = healthy.astype({'cik': int})
healthy = healthy.astype({'cik': str})

# cast dates to datetime
failed['period_of_report'] = pd.to_datetime(failed['period_of_report'])
failed['filing_date'] = pd.to_datetime(failed['filing_date'])
healthy['period_of_report'] = pd.to_datetime(healthy['period_of_report'])
failure_years['date_of_failure'] = pd.to_datetime(failure_years['date_of_failure'])

# add year variable (used in sampling code)
failed['year'] = failed['period_of_report'].apply(lambda x: x.year)
healthy['year'] = healthy['period_of_report'].apply(lambda x: x.year)
failed = failed.drop_duplicates()
healthy = healthy.drop_duplicates()


def label(row):
    """
    :param row: row from failed df
    :return: 1 if company filed for bankruptcy within 1 year of the filing date of the report, 0 otherwise
    """
    # retain cik
    cik = row['cik']
    # extract date of failure
    date_of_failure = failure_years[failure_years['cik'] == cik]['date_of_failure'].iloc[0]

    # check if the report was filed after the bankruptcy filing
    if date_of_failure < row['filing_date']:
        return -1
    # check if the report was filed within one year of the bankruptcy filing
    elif relativedelta(date_of_failure, row['filing_date']).years == 0:
        return 1
    # all remaining cases
    else:
        return 0


# compute label
failed['label'] = failed.apply(label, axis=1)
healthy['label'] = 0
# drop reports filed after bankruptcy filing
failed = failed.drop(failed[failed['label'] < 0].index)

# if multiple filings from same year (e.g. fund or changing date) retain first
failed = failed.drop_duplicates(['cik', 'year', 'indicator'])
healthy = healthy.drop_duplicates(['cik', 'year'])

# order cols
cols = ['cik', 'period_of_report', 'item_7', 'year', 'filing_date', 'label']
healthy = healthy[cols]
failed = failed[cols]

# store the exact failure date
failed['failure_date'] = failed['cik'].apply(
    lambda x: failure_years[failure_years['cik'] == x]['date_of_failure'].iloc[0])

# reset the index and store
failed.reset_index(drop=True).to_csv(path + 'failed_transformers.csv')
healthy.reset_index(drop=True).to_csv(path + 'healthy_transformers.csv')
