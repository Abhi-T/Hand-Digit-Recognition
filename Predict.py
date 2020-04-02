import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys,os

#loading the model
json_file=open('model-bw.json','r')
model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(model_json)
#load weights into new model
loaded_model.load_weights('model-bw.h5')
print('Loaded model from disk')

cap=cv2.VideoCapture(0)

#category dictionary
categories={0:'ZERO',1:'ONE',2:'TWO',3:'THREE',4:'FOUR',5:'FIVE'}

while True:
    _,frame=cap.read()
    #simulating the mirror image
    frame=cv2.flip(frame,1)

    #got this from collect-data.py
    #coordinates of the ROI
    x1=int(0.5 * frame.shape[1])
    y1=10
    x2=frame.shape[1]-10
    y2=int(0.5 * frame.shape[1])

    #drawing the roi
    #the increment/decrement by 10 is to compensate for bounding box
    cv2.rectangle(frame, (x1-1,y1-1),(x2+1,y2+1), (255,0,0), 1 )
    #extracting the Roi
    roi=frame[y1:y2,x1:x2]
    #resixing the roi so that it can be fed to the model for predictions
    roi=cv2.resize(roi, (64,64))
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _,test_image= cv2.threshold(roi, 120,255, cv2.THRESH_BINARY)
    cv2.imshow('test',test_image)
    #batch of 1
    result=loaded_model.predict(test_image.reshape(1,64,64,1))
    prediction={'ZERO':result[0][0],
                'ONE': result[0][1],
                'TWO': result[0][2],
                'THREE':result[0][3],
                'FOUR':result[0][4],
                'FIVE':result[0][5]}
    #sorting based on top prediction
    prediction=sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    #display prediction
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.imshow('frame',frame)

    interrupt=cv2.waitKey(10)
    if interrupt & 0xFF==27: #esc key
        break

cap.release()
cv2.destroyAllWindows()