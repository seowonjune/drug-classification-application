import urllib.request
import time
from PIL import Image
import cv2 

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
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#making edge
src = cv2.imread(image_name, cv2.IMREAD_COLOR)

canny = cv2.Canny(src, 310, 255)

cv2.imshow("canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()