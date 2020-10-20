
import numpy as np
import cv2
from matplotlib import pyplot as plt



def main():
    img = findFaces("test.jpg")
    displayImage(img)
        
def findFaces(imageFileName):
    #get the model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #read the image
    img = cv2.imread(imageFileName)
    #gray it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Find the faxes using the model
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #draw green box around faces
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    return img

def displayImage(img):
    #to make a resizeable window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #show the image in the window
    cv2.imshow('image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()