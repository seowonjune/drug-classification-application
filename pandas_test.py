# import pandas as pd

# pd.set_option('display.max_colwidth', -1) #prvent url cutted

# drug_url = pd.read_excel('api.xlsx', usecols='F')

# #for i in range(4, 6):
# a= drug_url.iloc[23928] #in programm 1 is 3 in excel file
# print(a)

import urllib.request
import time
import os
from PIL import Image
import cv2 
import pandas as pd
import shutil

pd.set_option('display.max_colwidth', -1) #prvent url cutted

drug_url = pd.read_excel('api.xlsx')

a = drug_url.iloc[0, 1] #in programm 1 is 3 in excel file
number = drug_url.iloc[0,0]
url = drug_url.iloc[0, 5]

base_dir = os.getcwd()

path = os.path.join(base_dir, str(number))

os.mkdir(path)

# os.rename(path, os.path.join(base_dir, b))


drug = pd.read_excel('api.xlsx')

image_name = drug.iloc[1, 1] + ".jpg"
url = drug.iloc[1, 5]

urllib.request.urlretrieve(url, image_name)
print(image_name + " dwonloading complete")

shutil.move(os.getcwd + '\\' + image_name, path + '\\' + image_name)
print( image_name + "moved")