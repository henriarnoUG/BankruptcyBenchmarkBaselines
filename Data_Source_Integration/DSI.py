# IMPORTS
import pandas as pd
from os import listdir
from datetime import datetime
import random


class DSI:
    """
    Class that is used to integrate the lopucki BRD case table and the EDGAR-CORPUS data sources.
    """

    def __init__(self, LOCATION_LOPUCKI, LOCATION_EDGAR):
        """
        :param LOCATION_LOPUCKI: location of the lopucki BRD case table
        :param LOCATION_EDGAR: location of the edgar corpus
        """
        self.location_lopucki = LOCATION_LOPUCKI
        self.location_edgar = LOCATION_EDGAR

    def integrate(self, begin_year, end_year):
        """
        :param begin_year: First year of filings to account for
        :param end_year: Last year of filings to account for
        :return: failed (healthy): a dataframe containing all filings from companies (not) present in the
        lopucki BRD case table.
        """

        # read in lopucki BRD
        lopucki = pd.read_csv(self.location_lopucki, encoding='unicode_escape')
        # select relevant variables
        BRD = lopucki[['NameCorp', 'CikBefore', 'CikEmerging', 'Date10k1Before', 'Date10k2Before',
                       'Date10k3Before', 'Date10kBefore', 'Date10kDuring']]

        # store the CIKs of companies in the BRD
        # also remove nan values and cast to string without decimals
        cik_list = list(BRD["CikBefore"])
        cik_list = [x for x in cik_list if str(x) != 'nan']
        cik_list = [str(int(x)) for x in cik_list]

        # for each filing, we store the information required for our models in a dataframe
        healthy_frame = pd.DataFrame(["cik", "period_of_report", "item_7", "filing_date", "indicator"])
        failure_frame = pd.DataFrame(["cik", "period_of_report", "item_7", "filing_date", "indicator"])

        # loop over each year
        for year in listdir(self.location_edgar):
            if (int(year) >= begin_year) & (int(year) <= end_year):
                pass
            else:
                continue
            # Keep track of the progress
            print('Reading files of year: ' + year)

            # store path to all 10-k filings from a specific year
            path_year = self.location_edgar + "/" + year

            # loop over each 10-k report in the current year
            for file in listdir(path_year):

                # read in each file and select relevant variables
                path_file = path_year + "/" + file
                filing = pd.read_json(path_file, typ='Series')
                filing = filing[['cik', 'period_of_report', 'item_7', 'filing_date']]

                # failed companies
                if filing['cik'] in cik_list:
                    # compute and add indicator
                    # this is NOT used in order to construct the label but purely informative
                    indicator = self.compute_indicator(BRD, filing)
                    filing = filing.append(pd.Series({'indicator': indicator}))
                    # add filing to df
                    failure_frame = failure_frame.append(filing, ignore_index=True)

                # healthy companies
                else:
                    # add indicator
                    # this is NOT used in order to construct the label but purely informative
                    filing = filing.append(pd.Series({'indicator': 0}))
                    # add filing to df
                    healthy_frame = healthy_frame.append(filing, ignore_index=True)

        # cast CIK to string
        failure_frame = failure_frame.astype({'cik': str})
        healthy_frame = healthy_frame.astype({'cik': str})

        # drop first rows with all missing values and order cols
        failure_frame = failure_frame[4:][["cik", "period_of_report", "item_7", "filing_date", "indicator"]]
        healthy_frame = healthy_frame[4:][["cik", "period_of_report", "item_7", "filing_date", "indicator"]]

        # return resulting dataframes
        return failure_frame, healthy_frame

    def compute_indicator(self, BRD, filing):
        """
        :param BRD: The lopucki BRD case table
        :param filing: the filing to process
        :return: an indicator that that shows if the report was the 10k1Before, 10k2Before or the 10k3Before
        according to the definition of the BRD case table (look at the lopucki BRD case table protocols for reference)
        """

        # extract the date of the 10-k reports 1, 2, 3 years before filing for bankruptcy
        date_1 = BRD[BRD['CikBefore'] == float(filing['cik'])].iloc[0]['Date10k1Before']
        date_2 = BRD[BRD['CikBefore'] == float(filing['cik'])].iloc[0]['Date10k2Before']
        date_3 = BRD[BRD['CikBefore'] == float(filing['cik'])].iloc[0]['Date10k3Before']
        # cast the dates to datetime
        date_1 = self.cast_to_date(date_1)
        date_2 = self.cast_to_date(date_2)
        date_3 = self.cast_to_date(date_3)

        # extract date of current filing
        date_filing = filing['period_of_report']
        date_filing = datetime.strptime(date_filing, '%Y-%m-%d')

        # return indicator
        if date_filing == date_1:
            indicator = 1
        elif date_filing == date_2:
            indicator = 2
        elif date_filing == date_3:
            indicator = 3
        else:
            indicator = 0

        return indicator

    def cast_to_date(self, date):
        """
        :param date: date as string to be cast to datetime
        :return: date as datetime (or none if input was no date)
        """
        try:
            result = datetime.strptime(date, '%m/%d/%Y')
        except:
            result = None
        return result

    def store_BRD_years (self):
        """
        :return: This function reads the lopucki BRD case table and extracts the exact dates on which companies
        filed for bankruptcy. This is returned as a pandas dataframe along with the cik and the year of failure.
        """

        # read in lopucki BRD
        lopucki = pd.read_csv(self.location_lopucki, encoding='unicode_escape')
        # select relevant variables
        BRD = lopucki[['CikBefore', 'DateFiled']]

        # cast cik to string and drop nan values
        BRD = BRD.astype({'CikBefore': str})
        BRD = BRD[BRD['CikBefore'] != 'nan']
        BRD = BRD.astype({'CikBefore': float})
        BRD['cik'] = BRD['CikBefore'].apply(lambda x: str(int(x)))

        # cast date to datetime and compute year
        BRD['date_of_failure'] = pd.to_datetime(BRD['DateFiled'])
        BRD['year_of_failure'] = BRD['date_of_failure'].apply(lambda x: x.year)

        # order cols and reset index
        BRD = BRD[['cik', 'year_of_failure', 'date_of_failure']]
        BRD = BRD.reset_index(drop=True)

        # return result
        return BRD

