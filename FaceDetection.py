import numpy as np
import cv2
from os import path

def main():
    selection = selectMethod()

    # Load the cascade
    front_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

    if(selection == 1):
        findFacesInImage(front_face_cascade, profile_face_cascade)
    else:
        findFacesInVideo(front_face_cascade, profile_face_cascade)


def findFacesInVideo(front_face_cascade, profile_face_cascade):
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

def findFacesInImage(front_face_cascade, profile_face_cascade):
    imageName = getImageName()
    image = cv2.imread(imageName,1)
    findFaces(image, front_face_cascade, profile_face_cascade)
    displayImage(image)

def selectMethod():
    print("Face Detection")
    print("Press 1 to detect faces in an image or 2 to detect faces in a live video")
    inp = int(input("Your selection: "))

    while(inp!=1 and inp!=2):
        print("Incorrect input!")
        print("Press 1 to detect faces in an image or 2 to detect faces in a video")
        inp = int(input("Your selection: "))
    return inp

def getImageName():
    print("Enter the name of the image (dont forget .jpg)")
    name = input("Image name: ")
    if(path.exists(name)==False):
        print("This image doesn't exist")
        exit(1)
    return name


def findFaces(image, front_face_cascade, profile_face_cascade):    
    #gray it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Find the faxes using the model
    front_faces = front_face_cascade.detectMultiScale(gray, 1.3, 4, 0, (64,64))
    left_profile_faces = profile_face_cascade.detectMultiScale(gray, 1.3, 4, 0, (64,64))
    flipped = cv2.flip(gray, 1)
    right_profile_faces = profile_face_cascade.detectMultiScale(flipped, 1.3, 4, 0, (64,64))
   

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