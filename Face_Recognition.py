"""
Created on Tuesday, 03/08/2021

@author: Hareesh Bhittam
"""

import  cv2

import  numpy as np

face_classifier = cv2.CascadeClassifier('D:/Projects/Face/haarcascade_frontalface_default.xml')

def face_extract(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

        return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extract(frame) is not None:
        count +=1
        face = cv2.resize(face_extract(frame),(350,350))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'D:/Projects/Face/TestData/users'+ str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(250,250),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0),2)

        cv2.imshow("Face Cropper",face)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1)==13 or count== 150:
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete!!')


    