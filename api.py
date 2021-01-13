import urllib.request
import time
import os
from PIL import Image
import cv2 

#loading execl file

#appointing url & name of image
url = "https://nedrug.mfds.go.kr/pbp/cmn/itemImageDownload/1Muwq7fAuBq"
image_name = "test.jpg"

#measuring time
start = time.time()

#downloading image
urllib.request.urlretrieve(url, image_name)

#printing time
print(time.time()- start)

print(image_name + " dwonloading complete")

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

#saving - made edge - jpg
# path = r'C:\Users\swj35\OneDrive\바탕 화면\drug-classification-application\edge'
# #image_name_path = image_name + "_path"
# cv2.imwrite(os.path.join(path , "test_edge.jpg"), canny)

cv2.imwrite('test_edge.jpg', canny)

print("test_edge.jpg download compeleted")