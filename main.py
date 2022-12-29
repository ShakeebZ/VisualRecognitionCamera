import sys
import cv2
import face_recognition
import numpy
import os
from time import sleep
from threading import Thread
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector

#global takingPhoto

def main():
    names, encodings = initialTraining()
    input = UserInput()
    if input == "1":
        CameraProgram1(names, encodings)
    elif input == "2":
        CameraProgram2()
    else:
        print("\nInvalid Input: Please enter an integer.\n")
        UserInput()


def initialTraining():
    path = sys.path[0] + "\TrainingFaces" + "\\"
    Names = []
    Encodings = []
    
    for __ in os.listdir(path):
        imageName = __.split(".")
        Names.append(imageName[0])
        image = face_recognition.load_image_file(path + __)
        Encodings.append(face_recognition.face_encodings(image)[0])
        
    print("Initial Training Completed!")
    
    return Names, Encodings


def CameraProgram1(Names, Encodings):
    video_capture = cv2.VideoCapture(0)
    
    #process_frame = True
    
    while(True):
        isReading, frame = video_capture.read() # read() outputs a value determining whether the operation ran succesfully and an image or frame
        small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25) #Resizing to 1/4 size for faster facial recognition processing
        small_frame = small_frame[:,:,::-1] #Converting from BGR(OpenCV uses this) to RGB(facial_recognition uses this)
        
        if True:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            for current_face in face_encodings:
                isPresent = face_recognition.compare_faces(Encodings, current_face, tolerance = 0.6)
                
                euclidean_distances = face_recognition.face_distance(Encodings, current_face) #Compare euclidean face distances of known faces to a face in the image
                best_match = numpy.argmin(euclidean_distances)
                
                if isPresent[best_match]:
                    face_name = Names[best_match]
                else: face_name = "Unknown"
            
        #process_frame = not process_frame

        faces = zip(face_locations, face_encodings)
        for (top, right, bottom, left), image_face_encodings in faces:
            #Scaling back to normal size
            top *= 4
            right *= 4
            left *= 4
            bottom *= 4
                
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) #Border around face
            cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 255, 0), cv2.FILLED) #Filled in rectangle below face for name
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, face_name, (left+6, bottom - 6), font, 1.0, (255, 255, 255), 1) #Placing text below face
        
        
        cv2.imshow("Camera Program #1", frame) #imshow takes in a string parameter to name the feed and an image or frame
        
        #NEED TO FIX THIS, camera won't close when q is pressed: Error is located in 0xFF == ('q')
        
        if cv2.waitKey(1) & 0xFF ==('q'): # Pressing q will exit the while loop and stop reading input from the camera
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def CameraProgram2():
    video_capture = cv2.VideoCapture(0)
    faceDetector = FaceDetector()
    handDetector = HandDetector(detectionCon=0.8, maxHands=1)
    takingPhoto = [False]
    
    print("Camera Program #2 Loaded. To take a photo, please make a peace symbol.\n")
    
    while True:
        isReading, frame = video_capture.read()
        if isReading:
            hands, image = handDetector.findHands(frame) #FlipType is default set to true, if hands are being recognized incorrectly then set to false as a paremeter
            #Every h in hands contains a dictionary {lmList, bbox, center, type} for a hand
                        
            if hands: #if a hand is detected
                hand1 = hands[0]
                lmList1 = hand1["lmList"] #List of 21 points of interest on the hand
                bbox1 = hand1["bbox"] #Bounding Box Information (x, y, w, h) w = width, h = height
                center1 = hand1["center"] #center of the hand, cx, cy
                handType1 = hand1["type"] #Returns Left or Right
                finger1 = handDetector.fingersUp(hand1)
            
                if (finger1 == [0, 1, 1, 0, 0]) and (takingPhoto[0] == False): #If user is holding a peace symbol
                    takingPhoto[0] = True
                    photoThread = Thread(target = takePhoto, args=(video_capture,takingPhoto))
                    photoThread.start()
            
            cv2.imshow("Camera Program #2", image)
            cv2.waitKey(1) #1ms
            
        if cv2.waitKey(1) & 0xFF ==('q'): # Pressing q will exit the while loop and stop reading input from the camera
            break
            
    video_capture.release()
    cv2.destroyAllWindows()
    
def takePhoto(vid, photo):
    #path = sys.path[0] + "\Photos" + "\\"
    print("Taking Photo in 5 seconds.")
    for x in range(5,0,-1):
        print(x)
        sleep(1)
    print("Taking Photo.")
    isReading, img = vid.read()
    cv2.imwrite('image.jpg', img)
    photo[0] = False

def UserInput():
    choice = input(
        "\nWhat would you like to do?\n1)Run Camera Program #1 (Made Using OpenCV)\n2)Run Camera Program #2 (Made Using CVZone)\n3)Train a new face\n")
    if choice.isnumeric():
        return choice
    else:
        print("\nInvalid Input: Please enter an integer.\n")
        UserInput()


if __name__ == "__main__":
    main()