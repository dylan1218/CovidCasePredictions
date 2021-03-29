import datetime

import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from collections import namedtuple
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import warnings
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.optimize import curve_fit
from pandas.plotting import autocorrelation_plot
warnings.filterwarnings("ignore")
plt.ioff()


class HackathonModels:
    def __init__ (self, df, region): #df as pandas df, region as string 
        self.df = df #must pass either reg_countries, reg_states, or reg_counties dfs
        self.region = region #"Country, State, County"

    def reset(self):
        dic = vars(self)
        for i in dic.keys():
            dic[i] = 0

    
    
    def getcountrydf(self, fields): #fields as type list
        #example usage: GetCountryDF("Denmark", ['date', 'new_cases']), or GetCountryDF('United States, ['date', 'total_cases])
        countries = self.df.loc[self.df['fips'] == 0]
        countries[['total_cases', 'new_cases']] = countries[['total_cases', 'new_cases']].bfill().astype('float32')
        df_country = countries.loc[countries['country'] == self.region]
        reg_df = df_country[fields]
        reg_df.set_index('date', inplace=True)
        return reg_df
        
    def GetStateFIPS(self): 
        stateFipMapping = {
        'CA': 6000,
        'AK': 2000,
        'AL': 1000,
        'AR': 5000,
        'AZ': 4000,
        'CO': 8000,
        'CT': 9000,
        'DC': 11000,
        'DE': 10000,
        'FL': 12000,
        'GA': 13000,
        'HI': 15000,
        'IA': 19000,
        'ID': 16000,
        'IL': 17000,
        'IN': 18000,
        'KS': 20000,
        'KY': 21000,
        'LA': 22000,
        'MA': 25000,
        'MD': 24000,
        'ME': 23000,
        'MI': 26000,
        'MN': 27000,
        'MO': 29000,
        'MS': 28000,
        'MT': 30000,
        'NC': 37000,
        'ND': 38000,
        'NE': 31000,
        'NH': 33000,
        'NJ': 34000,
        'NM': 35000,
        'NV': 32000,
        'NY': 36000,
        'OH': 39000,
        'OK': 40000,
        'OR': 41000,
        'PA': 42000,
        'RI': 44000,
        'SC': 45000,
        'SD': 46000,
        'TN': 47000,
        'TX': 48000,
        'UT': 49000,
        'VA': 51000,
        'VT': 50000,
        'WA': 53000,
        'WI': 55000,
        'WV': 54000,
        'WY': 56000
        }
        return stateFipMapping.get(self.region)

    def getstatedf(self, fields): #fields as type list
        #example usage: GetStateDF(['date', 'new_cases'])
        states = self.df.loc[(self.df['fips'] % 1000 == 0) & (self.df['fips'] != 0)]
        idx = states.loc[states['county'] == "Grand Princess Cruise Ship"].index
        states.drop(idx, inplace=True)
        states[['total_cases', 'new_cases']] = states[['total_cases', 'new_cases']].bfill().astype('int')
        df_state = states.loc[states['fips'] == self.GetStateFIPS()] #call method below
        reg_df = df_state[fields]
        reg_df.set_index('date', inplace=True)
        return reg_df

    #to change to private method
    def SaveGraphModelDict(self, filename, docname, df):
        script_dir = os.path.dirname("__file__")
        results_dir = os.path.join(script_dir, 'Models/' + filename + "/")
        sample_file_name = filename + " " + docname
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        df.plot()
        plt.title("Sarimax for " + filename + " " + docname)
        if os.path.isfile(results_dir + sample_file_name):
            os.remove(results_dir + sample_file_name)    
        
        plt.savefig(results_dir + sample_file_name)
        plt.close()
    
    #to change to private method
    def SaveACP(self, filename, docname, df):
        script_dir = os.path.dirname("__file__")
        results_dir = os.path.join(script_dir, 'Models/' + filename + "/")
        sample_file_name = "ACF " + filename + " " + docname
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        autocorrelation_plot(df)
        plt.title("ACF for " + filename + " " + docname)
        if os.path.isfile(results_dir + sample_file_name):
            os.remove(results_dir + sample_file_name)
        plt.savefig(results_dir + sample_file_name)
        plt.close()

    #to change to private method
    def generateFilePath(self): #creates directory based off of base and region
        script_dir = os.path.dirname("__file__") #no base var needed with this approach, gets current file directory
        results_dir = os.path.join(script_dir,"Models/" +  self.region)        
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        return results_dir

    #to change to private method            
    def create_model(self, df, modelName):
        #fits ARIMA order and returns the most optimized order for use in SARIMAX
        saveDirectory = self.generateFilePath()
        stepwise_fit = auto_arima(df, start_p = 1, start_q = 1, max_p = 5, max_q = 5, seasonal=False,max_order=None,maxiter=500, error_action='ignore', suppress_warnings='True', stepwise=True)
        #fit_orders[country] = stepwise_fit.order instead of using the fit_orders df returning the order in function
        with open(saveDirectory + "/arima_summary" + modelName + ".txt", 'w') as f:
            f.write(str(stepwise_fit.summary()))

        #Set the minimum value of p or q to 2 if either is 0  
#        if stepwise_fit.order[2] == 0:
#            tupleList = list(stepwise_fit.order)
#            tupleList[2] = 2
#            stepwise_fit.order = tuple(tupleList)        

#        if stepwise_fit.order[0] == 0:
#            tupleList = list(stepwise_fit.order)
#            tupleList[0] = 2
#            stepwise_fit.order = tuple(tupleList)        
        return stepwise_fit.order

    def getPolynomialFit(self, df, predictDate):
        def polynomialExogenousRegressor(numpydfcolumn, extrapStepCount):
            y = numpydfcolumn
            x = list(range(0, len(y)))
            z = np.polyfit(x, y, 4)
            f = np.poly1d(z) 
            extrapList = [f(val) for val in range(0, len(y)+extrapStepCount)]
            return extrapList
                
        
        timeDeltaObjectEnd = datetime.datetime.strptime(predictDate, "%Y-%m-%d")
        timeDeltaObjectStart = df.index[len(df)-1] #gets last date in df var
        #delta between end and start
        dateTimeDelta = timeDeltaObjectEnd - timeDeltaObjectStart
        dateTimeDeltaInt = dateTimeDelta.days

        exogdates = pd.date_range(start=df.index[0], end=(df.index[len(df)-1]+dateTimeDelta)) #creates ranges of dates from start-end+delta
        exogDF = pd.DataFrame(exogdates, columns=["date"]).set_index("date")
        exogDF['polyexogvals'] = polynomialExogenousRegressor(df.to_numpy().flatten(),dateTimeDeltaInt)
        exogDF = sm.add_constant(exogDF.loc[:exogDF.index[len(exogDF)-1].strftime("%Y-%m-%d"), "polyexogvals"])
        return exogDF

    def linear_predict(self, df, predictDate):
        timeDeltaObjectEnd = datetime.datetime.strptime(predictDate, "%Y-%m-%d")
        timeDeltaObjectStart = df.index[len(df)-1] #gets last date in df var
        dateTimeDelta = timeDeltaObjectEnd - timeDeltaObjectStart
        #dateTimeDeltaInt = dateTimeDelta.days
        exogdates = pd.date_range(start=df.index[0], end=(df.index[len(df)-1]+dateTimeDelta)) #creates ranges of dates from start-end+delta
        predictDF = pd.DataFrame(exogdates, columns=["date"]).set_index("date")
        X_predict = pd.DataFrame(exogdates, columns=["date"]).set_index("date")
        X_predict = X_predict.index.map(datetime.datetime.toordinal).values.reshape(-1, 1)
        #y = df[['total_cases']]
        y = df.total_cases.values.reshape(-1, 1)
        #df['date1'] = df.index
        #X = df[['date1']]
        X = df.index.map(datetime.datetime.toordinal).values.reshape(-1, 1)
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        y_predict = model.predict(X_predict)
        predictDF['OLS linear prediction'] = y_predict 
        return predictDF


    def getSigmoidFit(self, df, predictDate):
        #https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
        #5pl may be better fit https://stats.stackexchange.com/questions/439362/trouble-optimizing-five-parameter-logistic-5pl-standard-curve-for-elisa-data-u
        def sigmoidExogenousRegressor(numpydfcolumn, extrapStepCount):
            def sigmoid(x, L ,x0, k, b):
                y = L / (1 + np.exp(-k*(x-x0)))+b
                return (y)                    
            ydata = numpydfcolumn #df.to_numpy().flatten()
            xdata = list(range(0, len(ydata)))             
            p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess
            popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='lm',  maxfev=10000)
            x = np.linspace(0, len(xdata)+extrapStepCount, len(xdata)+extrapStepCount) #1000000
            y = sigmoid(x, *popt)
            return y
        
        timeDeltaObjectEnd = datetime.datetime.strptime(predictDate, "%Y-%m-%d")
        timeDeltaObjectStart = df.index[len(df)-1] #gets last date in df var
        #delta between end and start
        dateTimeDelta = timeDeltaObjectEnd - timeDeltaObjectStart
        dateTimeDeltaInt = dateTimeDelta.days

        exogdates = pd.date_range(start=df.index[0], end=(df.index[len(df)-1]+dateTimeDelta)) #creates ranges of dates from start-end+delta
        exogDF = pd.DataFrame(exogdates, columns=["date"]).set_index("date")
        exogDF['polyexogvals'] = sigmoidExogenousRegressor(df.to_numpy().flatten(),dateTimeDeltaInt)
        exogDF = sm.add_constant(exogDF.loc[:exogDF.index[len(exogDF)-1].strftime("%Y-%m-%d"), "polyexogvals"])
        return exogDF        

    def run_sarimax(self, df, predictDate):
        #Example run_sarimax_exog(GetCountryDF("Denmark", ['date', new_cases']),"Denmark", "2020-07-2020")
        modelDict = {}
        ARIMAorder = self.create_model(df, "ARIMA")
        
        model = SARIMAX(
            df,
            order=ARIMAorder, 
            #trend="t",
            enforce_stationarity=False, 
            enforce_invertibility=False, 
            seasonal_order=(0, 0, 0, 0)
            #freq="D"
            )             
        result = model.fit(disp=False)
        #a=result.aic
        

        modelDict[predictDate] = ARIMAorder
        preds_sarimax = result.predict(
                start=df.index[0].strftime("%Y-%m-%d"), #df.index[0] was string
                end=predictDate, 
                typ='levels' #linear vs levels 
                #dynamic=True
                )
        df_preds_sarimax = pd.DataFrame(preds_sarimax, columns=["new_cases_predict"])
        df_preds_sarimax = df_preds_sarimax
        compareDF = df_preds_sarimax.merge(df, how='left', left_index=True, right_index=True)    
        self.SaveGraphModelDict(self.region,"SARIMAX",compareDF)   #can remove if errors
        return compareDF, modelDict  
            
    def run_sarimax_exog(self, df, exogdf, predictDate):
        #Example run_sarimax_exog(GetCountryDF("Denmark", ['date', new_cases']),exogdf,"Denmark", "2020-07-2020")
        #Output is a 2 sized array where run_sarimax_exog[0] is a dataframe of new_cases and new_predict
        modelDict = {}
        ARIMAorder = self.create_model(df, "ARIMALOG")
        dateTimeLastDateDF = df.index[len(df)-1]
        strLastDateDF = dateTimeLastDateDF.strftime("%Y-%m-%d") 
        strLastDateDF1 = (dateTimeLastDateDF + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            model = SARIMAX(
                df,
                exog=exogdf.loc[:strLastDateDF],
                order=ARIMAorder, 
                #trend="t",
                enforce_stationarity=False, 
                enforce_invertibility=False, 
                seasonal_order=(0, 0, 0, 0),
                freq="D"
                )             
            result = model.fit(disp=False)
            a=result.aic
        except Exception as e:
            #traceback.print_stack()
            print(e)

        modelDict[predictDate] = ARIMAorder
        preds_sarimax = result.predict(
                df.index[0].strftime("%Y-%m-%d"), #was a fixed string df.index[0].strftime("%Y-%m-%d") 
                predictDate, 
                typ='levels', #linear vs levels 
                exog=exogdf.loc[strLastDateDF1:]
                #dynamic=True
                )
        df_preds_sarimax = pd.DataFrame(preds_sarimax, columns=["new_cases_predict"])
        df_preds_sarimax = df_preds_sarimax
        compareDF = df_preds_sarimax.merge(df, how='left', left_index=True, right_index=True)
        self.SaveGraphModelDict(self.region,"SARIMAXEXOG",compareDF)       
        return compareDF, modelDict  
        
    def run_VAR_MAVGStringency(self, df, predictDate):
        #requires stringency index and new_cases, to refactor to allow total_cases later
        if 'stringency_index' not in df.columns:
            print("No stringency index found in given DF")
            return

        #Helper function to inverse the diff normalization
        def invert_transformation(df_diff, df_orig, second_diff=False):
            """Revert back the differencing to get the forecast to original scale."""
            columns = df_diff.columns
            for col in columns:   
                #print(col)
                x = df_orig[col].iloc[0]
                x_diff = df_diff[col].iloc[1:]     
                df_diff[col] = np.r_[x, x_diff].cumsum().astype(int)
                #df_diff[col +"diff"] = df_diff[col].rolling(2).sum().shift(-1)
            return df_diff
        

        #this requires a DF with stringency index
        #Data cleaning
        countryDF = df.fillna(0).replace(to_replace=0, method='ffill') #replaces NA with 0, and then replaces 0's with the last non-zero value
        countryDF['MovingAvgStringency'] = countryDF['stringency_index'].rolling(window=21).mean() #rolling window of 3 weeks, can be changed
        countryDF.loc[countryDF.new_cases == 0, 'new_cases'] = 1 #VAR model doesn't accept 0 values
        countryDF.loc[countryDF.stringency_index == 0, 'stringency_index'] = 1 #VAR model doesn't accept 0 values
        countryDF = countryDF.drop('stringency_index', axis=1) #VAR model is processed with the MAVG of stringency
        countryDF = countryDF.fillna(1)
        #countryDF = countryDF.loc['2020-03-01':]
        df_orig = countryDF #Store the predifferenced DF to undifference before returning predictions
        
        #Data normalizations, currently performing one diff, but need an algo to determine if more than one is needed. To refactor later
        countryDF = countryDF.diff().dropna() 

        #Generate VAR model
        predictNOBs = predictDate #This var should be calculated based off of the predict date, to refactor.
        model = VAR(endog=countryDF)
        model.select_order(15)
        results = model.fit(maxlags=15,ic="fpe") #determines optimal lags based off of AIC
        lag_order = results.k_ar
        varPred = results.forecast(countryDF.values[-lag_order:], steps=predictNOBs)

        #Creates prediction dates dataframe +datetime.timedelta(days=1)
        predDates = pd.date_range(start=countryDF.index[len(countryDF)-1]+datetime.timedelta(days=1), end=(countryDF.index[len(countryDF)-1]+datetime.timedelta(days=predictNOBs)))
        dateDF = pd.DataFrame(predDates, columns=['date'])

        #Merges predictions 
        predDF = pd.DataFrame(varPred.reshape(-1,2), columns=['new_cases', 'MovingAvgStringency']) #was prediction after col name
        predDF['date'] = dateDF.values.flatten()
        predDF = predDF.set_index('date')
        predDF = predDF.append(countryDF).sort_index()
        predDF = invert_transformation(predDF, df_orig)
        NOBCasesPred = predDF[-predictNOBs:][['new_cases', 'MovingAvgStringency']]
        NOBCasesPred = NOBCasesPred.rename({'new_cases': 'new_cases_predict', 'MovingAvgStringency':'MovingAvgStringency_predict'}, axis=1)
        predDF = predDF.merge(NOBCasesPred, how='left', right_index=True, left_index=True)
        self.SaveGraphModelDict(self.region,"VAR",predDF)
        return predDF


    def run_sarimax_exog(self, df, exogdf, predictDate):
        #Example run_sarimax_exog(GetCountryDF("Denmark", ['date', new_cases']),exogdf,"Denmark", "2020-07-2020")
        #Output is a 2 sized array where run_sarimax_exog[0] is a dataframe of new_cases and new_predict
        modelDict = {}
        ARIMAorder = self.create_model(df, "ARIMALOG")
        dateTimeLastDateDF = df.index[len(df)-1]
        strLastDateDF = dateTimeLastDateDF.strftime("%Y-%m-%d") 
        strLastDateDF1 = (dateTimeLastDateDF + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            model = SARIMAX(
                df,
                exog=exogdf.loc[:strLastDateDF],
                order=ARIMAorder, 
                #trend="t",
                enforce_stationarity=False, 
                enforce_invertibility=False, 
                seasonal_order=(0, 0, 0, 0),
                freq="D"
                )             
            result = model.fit(disp=False)
            a=result.aic
        except Exception as e:
            #traceback.print_stack()
            print(e)

        modelDict[predictDate] = ARIMAorder
        preds_sarimax = result.predict(
                df.index[0].strftime("%Y-%m-%d"), #was a fixed string df.index[0]
                predictDate, 
                typ='levels', #linear vs levels 
                exog=exogdf.loc[strLastDateDF1:]
                #dynamic=True
                )
        df_preds_sarimax = pd.DataFrame(preds_sarimax, columns=["new_cases_predict"])
        df_preds_sarimax = df_preds_sarimax
        compareDF = df_preds_sarimax.merge(df, how='left', left_index=True, right_index=True) 
        self.SaveGraphModelDict(self.region,"SARIMAXEXOG",compareDF)      
        return compareDF, modelDict  

    def run_VAR_MAVGStringency_tc(self, df, predictDate):
        #requires stringency index and total_cases, to refactor to allow variability in fields later
        if 'stringency_index' not in df.columns:
            print("No stringency index found in given DF")
            return

        #Check if all stringency values are 0, and if so breaks out of the VAR model
        if (df['stringency_index'] == 0).all():
            print("All stringency values are null, exiting model")
            return

        #Helper function to inverse the diff normalization
        def invert_transformation(df_diff, df_orig, second_diff=False):
            """Revert back the differencing to get the forecast to original scale."""
            columns = df_diff.columns
            for col in columns:   
                print(col)
                x = df_orig[col].iloc[0]
                x_diff = df_diff[col].iloc[1:]     
                df_diff[col] = np.r_[x, x_diff].cumsum().astype(int)
                #df_diff[col +"diff"] = df_diff[col].rolling(2).sum().shift(-1)
            return df_diff


        #this requires a DF with stringency index
        #Data cleaning
        countryDF = df.fillna(0).replace(to_replace=0, method='ffill') #replaces NA with 0, and then replaces 0's with the last non-zero value
        countryDF['MovingAvgStringency'] = countryDF['stringency_index'].rolling(window=21).mean() #rolling window of 3 weeks, can be changed
        countryDF.loc[countryDF.total_cases == 0, 'total_cases'] = 1 #VAR model doesn't accept 0 values
        countryDF.loc[countryDF.stringency_index == 0, 'stringency_index'] = 1 #VAR model doesn't accept 0 values
        countryDF.loc[countryDF.stringency_index == 0, 'MovingAvgStringency'] = 1 #VAR model doesn't accept 0 values
        countryDF = countryDF.drop('stringency_index', axis=1) #VAR model is processed with the MAVG of stringency
        countryDF = countryDF.fillna(1)
        #countryDF = countryDF.loc['2020-03-01':]
        df_orig = countryDF #Store the predifferenced DF to undifference before returning predictions
        
        #Data normalizations, currently performing one diff, but need an algo to determine if more than one is needed. To refactor later
        countryDF = countryDF.diff().dropna() 
        #return countryDF

        #Generate VAR model
        predictNOBs = predictDate #This var should be calculated based off of the predict date, to refactor.
        model = VAR(endog=countryDF)
        model.select_order(15)
        results = model.fit(maxlags=15,ic="fpe") #determines optimal lags based off of AIC
        lag_order = results.k_ar
        #return countryDF
        varPred = results.forecast(countryDF.values[-lag_order:], steps=predictNOBs+1)

        #Creates prediction dates dataframe +datetime.timedelta(days=1)
        predDates = pd.date_range(start=countryDF.index[len(countryDF)-1]+datetime.timedelta(days=1), end=(countryDF.index[len(countryDF)-1]+datetime.timedelta(days=predictNOBs+1)))
        dateDF = pd.DataFrame(predDates, columns=['date'])

        #Merges predictions 
        predDF = pd.DataFrame(varPred.reshape(-1,2), columns=['total_cases', 'MovingAvgStringency']) #was prediction after col name
        predDF['date'] = dateDF.values.flatten()
        predDF = predDF.set_index('date')
        predDF = predDF.append(countryDF).sort_index()
        predDF = invert_transformation(predDF, df_orig)
        NOBCasesPred = predDF[-predictNOBs-1:][['total_cases', 'MovingAvgStringency']]
        NOBCasesPred = NOBCasesPred.rename({'total_cases': 'total_cases_predict', 'MovingAvgStringency':'MovingAvgStringency_predict'}, axis=1)
        predDF = predDF.merge(NOBCasesPred, how='left', right_index=True, left_index=True)
        self.SaveGraphModelDict(self.region,"VAR",predDF)
        return predDF

    #Doesn't work yet
    def GenerateComparisonDF(self, df, exogdf, predictDate):

        sarimaxDF = self.run_sarimax(df, predictDate)
        sarimaxExogDF = self.run_sarimax_exog(df, exogdf, predictDate)
        varDF = self.run_VAR_MAVGStringency_tc(df, predictDate)

        #Generates a df template with date index to append predictions to
        timeDeltaObjectEnd = datetime.datetime.strptime(predictDate, "%Y-%m-%d")
        timeDeltaObjectStart = df.index[len(df)-1] #gets last date in df var
        #delta between end and start
        dateTimeDelta = timeDeltaObjectEnd - timeDeltaObjectStart
        #dateTimeDeltaInt = dateTimeDelta.days
        compareDates = pd.date_range(start=df.index[0], end=(df.index[len(df)-1]+dateTimeDelta)) #creates ranges of dates from start-end+delta
        compareDates = pd.DataFrame(compareDates, columns=["date"]).set_index("date")

        compareDates['sarimax_predictions'] = sarimaxDF #sarimaxDF.new_cases_predict.values.flatten()
        compareDates['sarimaxExog_predictions'] = sarimaxExogDF #.new_cases_predict.flatten()
        compareDates['var_predictions'] = varDF #.new_cases_predict.flatten()

        
        return compareDates
