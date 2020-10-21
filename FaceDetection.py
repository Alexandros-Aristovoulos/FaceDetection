import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    #img = findFaces("test.jpg")
    #displayImage(img)

    # Load the cascade
    front_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    
    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)
    
    #check if camera is detected
    if(cap.isOpened() == False):
        print("Could not detect a camera")


    while cap.isOpened():
        # Read the frame
        _, image = cap.read()
        #find the faces
        image = findFaces(image, front_face_cascade, profile_face_cascade)

        #to make a resizeable window
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # Display
        cv2.imshow('Face Detection', image)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    # Release the VideoCapture object
    cap.release()


        
def findFaces(image, front_face_cascade, profile_face_cascade):    
    #gray it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Find the faxes using the model
    front_faces = front_face_cascade.detectMultiScale(gray, 1.3, 5)
    left_profile_faces = profile_face_cascade.detectMultiScale(gray, 1.3, 5)
    flipped = cv2.flip(gray, 1)
    right_profile_faces = profile_face_cascade.detectMultiScale(flipped, 1.3, 5)
   

    #Front Face
    for (x,y,w,h) in front_faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0), 2)
        
    #Left Faces
    for (x,y,w,h) in left_profile_faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0), 2)

    #Right Faces (we use the fliped image)
    for (x,y,w,h) in right_profile_faces:
        image = cv2.flip(image,1)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255), 2)               
        image = cv2.flip(image,1)

    return image

def displayImage(img):
    #to make a resizeable window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #show the image in the window
    cv2.imshow('image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()