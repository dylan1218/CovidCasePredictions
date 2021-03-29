import pandas as pd 
from functools import reduce
from pytrends.request import TrendReq

class MergeTrendsDFs:
    def __init__(self, regionIdentifier):
        self.regionIdentifier = regionIdentifier
    
    def getTrendsData(self, dateRange):
        #Search terms subjected to 5 term limit, 2d array and loop to solve this

        symptomsList2d = [
                            ['Fever', 'Chills', 'Shortness of breath', 'Difficulty breathing', 'Fatigue'],
                            ['Body aches', 'Headache','Loss of taste', 'Loss of smell', 'Sore throat'],
                            ['Congestion', 'Runny nose', 'Nausea', 'Vomiting']   
                        ]
        #Defined from CDC https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html
        #Note: English language symptoms only for right now. To evaluate how to handle non-english speaking countries
        
        #https://github.com/GeneralMills/pytrends/blob/master/examples/example.py
        pytrend = TrendReq(hl='en-US',timeout=(10,25), retries=10,  backoff_factor=1, tz=360) #set TrendReq object

        counter = 0 #Counter is used to create separate output files, could change to something more descriptive but works for now
        for symptomArray in symptomsList2d:
            pytrend.build_payload(
                kw_list=symptomArray,
                cat=0,
                timeframe=dateRange, #'2020-01-01 2020-07-12'
                geo=self.regionIdentifier, #To confirm which countries/states we want to test this approach with
                gprop='')
            counter = counter + 1
            interest_over_time_df = pytrend.interest_by_region('request') #call interest_by_region method
            data = pytrend.interest_over_time()
        
            data.to_csv("C:\\Users\\user\\Development\\Covid Hackathon\\TrendsWorkingOutput\\" + self.regionIdentifier + "-covid_trends" + str(counter) + ".csv", encoding='utf_8_sig')


    def dropHeader(self, headerName, dfArray):
        #clean dfs
        dfArrayFiltered = []
        for df in dfArray:
            df = df.drop(headerName, axis=1)
            dfArrayFiltered.append(df)
        return dfArrayFiltered

    def mergeOnDate(self, dfs):
        df_final = reduce(lambda left,right: pd.merge(left,right,on='date'), dfs)
        return df_final

