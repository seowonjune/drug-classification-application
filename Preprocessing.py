import urllib.request
import time
import os
from PIL import Image
import cv2 
import pandas as pd
import shutil

#loading execl file
pd.set_option('display.max_colwidth', -1) #prvent url cutted
drug = pd.read_excel('api.xlsx')
# drug_url = pd.read_excel('api.xlsx', usecols='F')
# drug_name = pd.read_excel('api.xlsx', usecols='B') #열 읽기, skiprows=[]는 행 읽기

#appointing url & name of image & using for
for i in range(0, 23828):
    number = drug.iloc[i,0]
    url = drug.iloc[i, 5]
    image_name_file = drug.iloc[i, 1]
    image_name = drug.iloc[i, 1] + ".jpg"

    #making image file with 품목번호
    base_dir = os.getcwd()
    path = os.path.join(base_dir, str(number))
    os.mkdir(path)

    #downloading image 
    urllib.request.urlretrieve(url, image_name)
    print(image_name_file + " dwonloading complete")
    shutil.move(os.getcwd + '\\' + image_name, path + '\\' + image_name)
    print( image_name_file + "moved")

    #open image using open_cv
    img_color = cv2.imread(image_name, cv2.IMREAD_COLOR)
    cv2.imshow(image_name, img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cutting image
    cut = img_color.copy()
    cut = img_color[30:340, 65:720]
    cv2.imshow(image_name, cut)

    cv2.imwrite('test_cut.jpg', cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #making edge
    canny = cv2.Canny(cut, 310, 255)

    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# saving - made edge - jpg
cv2.imwrite(os.path.join(path + '\\' , "test_edge.jpg"), canny)

cv2.imwrite('test_edge.jpg', canny)

print("test_edge.jpg download compeleted")

#OCR with Tesseract