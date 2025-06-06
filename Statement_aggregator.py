#  This code works with sec-api

import sec_api
import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict

class FinData:
    def __init__(self, tickers):
        self.apiKey = '79546d5c7538e63abcd6bcc3dd8d7992ef3df1608d09afa3859246318573f657'
        self.tickerNames = tickers
        self.genInfo = None

        self.mappingApi = sec_api.MappingApi(api_key=self.apiKey)
        self.queryApi = sec_api.QueryApi(api_key=self.apiKey)
        self.xbrlApi = sec_api.XbrlApi(api_key=self.apiKey)


        self.statements  = ['StatementsOfIncome', 'StatementsOfIncomeParenthetical', 'StatementsOfComprehensiveIncome', 'StatementsOfComprehensiveIncomeParenthetical', 'BalanceSheets', 'BalanceSheetsParenthetical', 
                            'StatementsOfCashFlows', 'StatementsOfCashFlowsParenthetical', 'StatementsOfShareholdersEquity', 'StatementsOfShareholdersEquityParenthetical']

    def getSummaryDF(self):
        """Turn the general info given by sec-api into a pandas DF"""
        info = defaultdict(list)

        for ticker in self.tickerNames:
            infoDict = self.mappingApi.resolve('ticker', ticker)
            for key, val in infoDict[0].items():
                info[key].append(val)

        self.genInfo = pd.DataFrame(data=info)

        return self.genInfo
    
    def _getfiling_info(self, ticker, startDate, endDate, filingType="10-Q"):
        """Exctracts how ever many 10Q statements are between the given dates"""

        querySent = 'ticker:' + ticker +' AND ' + 'filedAt:{' + startDate + ' TO ' + endDate + "} AND "+ f"formType:\"{filingType}\""

        query = {
        "query": { "query_string": { 
            "query": querySent
            } },
        "from": "0",
        "size": "100",
        "sort": [{ "filedAt": { "order": "desc" } }]
        }

        filings = self.queryApi.get_filings(query)
        filtered_filings = []
        for filing in filings['filings']:
            if filing['formType'] == filingType:
                filtered_filings.append(filing)

        statements = [self.xbrlApi.xbrl_to_json(accession_no=d['accessionNo']) for d in filtered_filings]

        # Remove the blocks, like FinancialInstrumentsDisclosureTextBlock and RevenueFromContractWithCustomerTextBlock, for example
        return statements
    
    def _organize_statements(self):
        """Organize the statement info given from the dictionaries into data frames for each statement"""

        main_statements = set(["BalanceSheets", "StatementsOfIncome", "StatementsOfCashFlows", "StatementsOfShareholdersEquity"])
        main_statements_dict = defaultdict(list)

        #   Look for BalanceSheets, StatementsOfIncome, StatementsOfCashFlows and StatementsOfShareholdersEquity. Collection them in a dictionary, so all the statement types are together
        statements = self._getfiling_info("WOOF", "2022-01-01", "2023-01-01")
        for statement_info_dict in statements:
            for section in statement_info_dict:
                if section in main_statements:
                    main_statements_dict[section].append(statement_info_dict[section])

        #   Create a DF where the rows are the accting items (even if not all quarters have them). Make the columns the dates. Fill in the df data spots with 0s (use DF.data = np.zeros(DF dimensions))
        statement_df_list = []
        for statement_type, statement_list in main_statements_dict.items():
            #   For each statement type like BalanceSheets, StatementOfIncome, etc, create a df to accomadate all of the segments 
            cols = []
            rows = []

            for statement in statement_list:
                print(statement)

        # return main_statements_dict


a = FinData(["TSLA", "AAPL"])

a._organize_statements()

""" Below is an improvement of the above code"""

import sec_api
import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict

class FinData:
    def __init__(self, tickers):
        self.apiKey = '79546d5c7538e63abcd6bcc3dd8d7992ef3df1608d09afa3859246318573f657'
        self.tickerNames = tickers
        self.genInfo = None

        self.mappingApi = sec_api.MappingApi(api_key=self.apiKey)
        self.queryApi = sec_api.QueryApi(api_key=self.apiKey)
        self.xbrlApi = sec_api.XbrlApi(api_key=self.apiKey)


        self.statements  = ['StatementsOfIncome', 'StatementsOfIncomeParenthetical', 'StatementsOfComprehensiveIncome', 'StatementsOfComprehensiveIncomeParenthetical', 'BalanceSheets', 'BalanceSheetsParenthetical', 
                            'StatementsOfCashFlows', 'StatementsOfCashFlowsParenthetical', 'StatementsOfShareholdersEquity', 'StatementsOfShareholdersEquityParenthetical']

    def getSummaryDF(self):
        """Turn the general info given by sec-api into a pandas DF"""
        info = defaultdict(list)

        for ticker in self.tickerNames:
            infoDict = self.mappingApi.resolve('ticker', ticker)
            for key, val in infoDict[0].items():
                info[key].append(val)

        self.genInfo = pd.DataFrame(data=info)

        return self.genInfo
    
    def _getfiling_info(self, ticker, startDate, endDate, filingType="10-Q"):
        """Exctracts how ever many 10Q statements are between the given dates"""

        querySent = 'ticker:' + ticker +' AND ' + 'filedAt:{' + startDate + ' TO ' + endDate + "} AND "+ f"formType:\"{filingType}\""

        query = {
        "query": { "query_string": { 
            "query": querySent
            } },
        "from": "0",
        "size": "100",
        "sort": [{ "filedAt": { "order": "desc" } }]
        }

        filings = self.queryApi.get_filings(query)
        filtered_filings = []
        for filing in filings['filings']:
            if filing['formType'] == filingType:
                filtered_filings.append(filing)

        statements = [self.xbrlApi.xbrl_to_json(accession_no=d['accessionNo']) for d in filtered_filings]

        # Remove the blocks, like FinancialInstrumentsDisclosureTextBlock and RevenueFromContractWithCustomerTextBlock, for example
        return statements
    
    def _organize_accting_info(self):
        """Once the statements are organized into a dict together..."""
        
    
    def _accting_item_info(self):
        """Organize the statement info given from the dictionaries into data frames for each statement"""

        main_statements = set(["BalanceSheets", "StatementsOfIncome", "StatementsOfCashFlows", "StatementsOfShareholdersEquity"])
        main_statements_dict = defaultdict(list)

        #   Look for BalanceSheets, StatementsOfIncome, StatementsOfCashFlows and StatementsOfShareholdersEquity. Collection them in a dictionary, so all the statement types are together
        statements = self._getfiling_info("PDM", "2022-01-01", "2023-01-01", filingType="10-K") 
        for statement_info_dict in statements:
            for section in statement_info_dict:
                if section in main_statements:
                    main_statements_dict[section].append(statement_info_dict[section])

        #   Create a DF where the rows are the accting items (even if not all quarters have them). Make the columns the dates. Fill in the df data spots with 0s (use DF.data = np.zeros(DF dimensions))
        statement_df_list = []
        for statement_type, statement_list in main_statements_dict.items():
            stmt_items_dict = {}
            print(f"-----------{statement_type}------------\n")
            #   For each statement type like BalanceSheets, StatementOfIncome, etc, create a df to accomadate all of the segments 
            for statement in statement_list:
                    #   Now loop through the accounting items and group them for each statement type.
                for accting_item, item_info in statement.items():
                    stmt_items_dict[accting_item] = item_info

            #   Now that the items are grouped, create the df rows (accting items) by making the date columns and placing the respective value with it.
            for item, info_list in stmt_items_dict.items():
                #   Grab data point thats not "Revenues" and place it into its respective accting item row and date col. Revenues will be in a seperate df, detailing the revenue from each segment
                print(f"{item}\n")
                for info_dict in info_list:
                    """IMPORTANT!!!!!! MOST OF THE INFO DICTS WITH A SEGMENT KEY ARE DUPLICATES THAT CAN BE IGNORED FOR NOW..."""
                    print(info_dict)

        # return main_statements_dict


a = FinData(["TSLA", "AAPL", "APLE", "PDM"])

a._accting_item_info()

#    THE BELOW CODE IS AN UPDATED VERSION OF THE CODE ABOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import sec_api
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

class FinData:
    def __init__(self, tickers):
        self.apiKey = '79546d5c7538e63abcd6bcc3dd8d7992ef3df1608d09afa3859246318573f657'
        self.tickerNames = tickers
        self.genInfo = None

        self.mappingApi = sec_api.MappingApi(api_key=self.apiKey)
        self.queryApi = sec_api.QueryApi(api_key=self.apiKey)
        self.xbrlApi = sec_api.XbrlApi(api_key=self.apiKey)


        self.statements  = ['StatementsOfIncome', 'StatementsOfIncomeParenthetical', 'StatementsOfComprehensiveIncome', 'StatementsOfComprehensiveIncomeParenthetical', 'BalanceSheets', 'BalanceSheetsParenthetical', 
                            'StatementsOfCashFlows', 'StatementsOfCashFlowsParenthetical', 'StatementsOfShareholdersEquity', 'StatementsOfShareholdersEquityParenthetical']

    def getSummaryDF(self):
        """Turn the general info given by sec-api into a pandas DF"""
        info = defaultdict(list)

        for ticker in self.tickerNames:
            infoDict = self.mappingApi.resolve('ticker', ticker)
            for key, val in infoDict[0].items():
                info[key].append(val)

        self.genInfo = pd.DataFrame(data=info)

        return self.genInfo
    
    def _getfiling_info(self, ticker, startDate, endDate, filingType="10-Q"):
        """Exctracts how ever many 10Q statements are between the given dates"""

        querySent = 'ticker:' + ticker +' AND ' + 'filedAt:{' + startDate + ' TO ' + endDate + "} AND "+ f"formType:\"{filingType}\""

        query = {
        "query": { "query_string": { 
            "query": querySent
            } },
        "from": "0",
        "size": "100",
        "sort": [{ "filedAt": { "order": "desc" } }]
        }

        filings = self.queryApi.get_filings(query)
        filtered_filings = []
        for filing in filings['filings']:
            if filing['formType'] == filingType:
                filtered_filings.append(filing)

        statements = [self.xbrlApi.xbrl_to_json(accession_no=d['accessionNo']) for d in filtered_filings]

        # Remove the blocks, like FinancialInstrumentsDisclosureTextBlock and RevenueFromContractWithCustomerTextBlock, for example
        return statements
    
    def _organize_accting_info(self):
        """Once the statements are organized into a dict together..."""
        
    
    def _accting_item_info(self, ticker, startDate, endDate, filingType="10-Q"):
        """Organize the statement info given from the dictionaries into data frames for each statement"""
        finished_stmnts = []

        main_statements = set(["BalanceSheets", "StatementsOfIncome", "StatementsOfCashFlows", "StatementsOfShareholdersEquity"])
        main_statements_dict = defaultdict(list)

        #   Look for BalanceSheets, StatementsOfIncome, StatementsOfCashFlows and StatementsOfShareholdersEquity. Collection them in a dictionary, so all the statement types are together
        statements = self._getfiling_info(ticker, startDate, endDate, filingType=filingType) 
        for statement_info_dict in statements:
            for section in statement_info_dict:
                if section in main_statements:
                    main_statements_dict[section].append(statement_info_dict[section])

        #   Create a DF where the rows are the accting items (even if not all quarters have them). Make the columns the dates. Fill in the df data spots with 0s (use DF.data = np.zeros(DF dimensions))

        for statement_type, statement_list in main_statements_dict.items():
            stmt_items_dict = {}
            print(f"-----------{statement_type}------------\n")
            #   For each statement type like BalanceSheets, StatementOfIncome, etc, create a df to accomadate all of the segments 
            for statement in statement_list:
                    #   Now loop through the accounting items and group them for each statement type.
                for accting_item, item_info in statement.items():
                    stmt_items_dict[accting_item] = item_info

            items_groups = []

            #   Now that the items are grouped, create the df rows (accting items) by making the date columns and placing the respective value with it.
            for item, info_list in stmt_items_dict.items():
                #   Grab data point thats not "Revenues" and place it into its respective accting item row and date col. Revenues will be in a seperate df, detailing the revenue from each segment

                #   Only one value associated with a specific date per accting item
                used_dates = set()

                for info_dict in info_list:
                    """IMPORTANT!!!!!! MOST OF THE INFO DICTS WITH A SEGMENT KEY ARE DUPLICATES THAT CAN BE IGNORED FOR NOW. THERES COPIES OF ITEM DATA THAT DOESNT HAVE SEGMENT IN THEIR INFO DICT. FILTER THESE OUT!"""
                    #   Skip the dicts that contain the key "segments"

                    if type(info_dict) == dict and "segment" not in info_dict.keys():
                        if "value" in info_dict.keys():
                            date = f"{info_dict['period']['startDate']}_{info_dict['period']['endDate']}" if len(info_dict["period"].keys()) == 2 else info_dict["period"]['instant']

                            if date not in used_dates:
                                value = float(info_dict["value"]) * 10**int(info_dict["decimals"]) if type(info_dict["decimals"]) == int else info_dict["value"]
                                items_groups.append((item, date, value))
                                used_dates.add(date)
                            else:
                                continue

            #   Organize the rows (items) and cols (dates).     MAKE SURE TO SORT THE DATES BEFORE THE BECOME THE COLS OF THE DF!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            cols = []
            rows = []
            used_cols = set()
            used_rows = set()

            for (item, date, value) in items_groups:
                if item not in used_rows:
                    rows.append(item)
                    used_rows.add(item)
                
                if date not in used_cols:
                    cols.append(date)
                    used_cols.add(date)

            #   Once the dates are in cols, sort the dates in ascending order
            cols = sorted(cols, key=lambda x: datetime.strptime(x.split("_")[0], "%Y-%m-%d"))

            #   Create the statement df
            stmnt_df = pd.DataFrame(index=rows, columns=cols, data=np.zeros((len(rows), len(cols))))
            
            #   Inject the values into their respective place
            for (item, date, value) in items_groups:
                stmnt_df.at[item, date] = value
            finished_stmnts.append(stmnt_df)
        
        return finished_stmnts



a = FinData(["TSLA", "AAPL", "APLE", "PDM"])
print(a._accting_item_info("APLE", "2020-01-01", "2023-01-01"))
