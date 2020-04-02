import os
import cv2

d='C:\\Users\\1000267332\\PycharmProjects\\OpenCV\\Practice\\Hand-Digit\\data\\train\\0'
newd='C:\\Users\\1000267332\\PycharmProjects\\OpenCV\\Practice\\Hand-Digit\\data\\train\\0\\new'

for walk in os.walk(d):
    for file in walk[2]:
        roi=cv2.imread(d+'/'+file)
        roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(roi,120,255, cv2.THRESH_BINARY)
        cv2.imwrite(newd + "/" + file, mask)