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

