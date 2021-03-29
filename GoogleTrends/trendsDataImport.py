from pytrends.request import TrendReq
from pandasMerge import MergeTrendsDFs
import pandas as pd 


regionIdentifier = 'US-NJ'
dfMergreClass = MergeTrendsDFs(regionIdentifier)

dfMergreClass.getTrendsData('2020-01-01 2020-07-12') #outputs into workingoutput folderpath


#Read exported CSV files
#First trends subarray
df1 = pd.read_csv("C:\\Users\\user\\Development\\Covid Hackathon\\TrendsWorkingOutput\\"+ regionIdentifier +"-covid_trends1.csv", engine='python', encoding = "utf-8-sig")
#Second trends subarray
df2 = pd.read_csv("C:\\Users\\user\\Development\\Covid Hackathon\\TrendsWorkingOutput\\"+ regionIdentifier +"-covid_trends2.csv", engine='python', encoding = "utf-8-sig")
#Third trends subarray
df3 = pd.read_csv("C:\\Users\\user\\Development\\Covid Hackathon\\TrendsWorkingOutput\\"+ regionIdentifier +"-covid_trends3.csv", engine='python', encoding = "utf-8-sig")

dfArray = [df1, df2, df3]

filteredDFs = dfMergreClass.dropHeader('isPartial', dfArray) #drops the isPartial header from the dfs in df array
#print(filteredDFs[0].head())

#Merges the filteredDFs into a single df
mergedDF = dfMergreClass.mergeOnDate(filteredDFs)

#exports the merged df
mergedDF.to_csv("C:\\Users\\user\\Development\\Covid Hackathon\\TrendsFinalOutput\\" + regionIdentifier + 'covid_trends.csv', encoding='utf_8_sig')

print(dfMergreClass.mergeOnDate(filteredDFs).head())





  

