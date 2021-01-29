import pandas as pd

pd.set_option('display.max_colwidth', -1) #prvent url cutted

drug_url = pd.read_excel('api.xlsx', usecols='F')

#for i in range(4, 6):
a= drug_url.iloc[23928] #in programm 1 is 3 in excel file
print(a)
