import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


basename='image/s7x223y835.jpg'
file_name = os.path.splitext(basename)[0]
img = cv.imread(basename)
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


#Threshold segmentation
ret,thresh1=cv.threshold(grayImage,90,255,cv.THRESH_BINARY)
cv.imwrite(file_name+'-BINARY.png', thresh1, [cv.IMWRITE_PNG_COMPRESSION, 0])  
ret,thresh2=cv.threshold(grayImage,90,255,cv.THRESH_TRUNC)  
cv.imwrite(file_name+'-thresh.png', thresh2, [cv.IMWRITE_PNG_COMPRESSION, 0])  
ret, thresh3 = cv.threshold(grayImage, 0, 255, cv.THRESH_OTSU)   
cv.imwrite(file_name+'-OTSU.png', thresh3, [cv.IMWRITE_PNG_COMPRESSION, 0]) 
ret, thresh4 = cv.threshold(grayImage, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
cv.imwrite(file_name+'-TRIANGLE.png', thresh4, [cv.IMWRITE_PNG_COMPRESSION, 0]) 
thresh5 = cv.adaptiveThreshold(grayImage, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
cv.imwrite(file_name+'-adaptive.png', thresh5, [cv.IMWRITE_PNG_COMPRESSION, 0]) 
#clustering
rows, cols = grayImage.shape[:]
data = grayImage.reshape((rows * cols, 1))
data = np.float32(data)
criteria = (cv.TERM_CRITERIA_EPS +
            cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS
#K-Means
compactness, labels, centers = cv.kmeans(data, 2, None, criteria, 10, flags)
dst = labels.reshape((grayImage.shape[0], grayImage.shape[1]))
for i in range(rows):
    for j in range(cols):
        if dst[i-1][j-1]==0:
            dst[i-1][j-1]=255
        else:
            dst[i-1][j-1]=0
cv.imwrite(file_name+'-kmeans.png', dst,[cv.IMWRITE_PNG_COMPRESSION, 0])
plt.imshow(thresh5, 'gray')
plt.show()  
cv.waitKey(0)
cv.destroyAllWindows()






