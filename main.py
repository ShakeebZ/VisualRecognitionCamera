import sys
import cv2
import face_recognition
import numpy
import os
from time import sleep
from threading import Thread
from threading import Event
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector

#global takingPhoto

def main():
    print("Welcome. This is a Simple Facial and Gesture Recognition program.")
    names, encodings = initialTraining()
    UserInput(names, encodings)
    
def camera(vid, done):
    while True:
        isReading, frame = vid.read()
        cv2.imshow("Camera Feed", frame)
        if (done.is_set()): print("Event was set");break
    
def trainFace(Names, Encodings):
    faceDetector = FaceDetector()
    video_capture = cv2.VideoCapture(0)
    done = Event()
    path = sys.path[0] + "\TrainingFaces" + "\\"
    trainingThread = Thread(target = takePhoto, args=(video_capture, done))
    firstName =""
    lastName = ""
    input("\nPlease press Enter when you are ready to begin.")
    while True:
        isReading, frame = video_capture.read()
        image, bbox = faceDetector.findFaces(frame)
        cv2.imshow("Training Feed", image)
        if (not done.is_set()):
            if (len(bbox)==1) and (not trainingThread.is_alive()):
                trainingThread.start()
            elif (len(bbox) > 1) and (not trainingThread.is_alive()):
                print("Multiple Faces detected. Restarting Training Process.\n")
                sleep(2)
            elif(len(bbox) == 0) and (not trainingThread.is_alive()):
                print("No Faces detected. Restarting Training Process.\n")
                sleep(2)
        elif (done.is_set()):
            firstName = input("Please enter your first name: ")
            lastName = input("Please enter your last name: ")
            cv2.imwrite(os.path.join(path,firstName + " " + lastName + ".jpg"), frame)
            Names.append(firstName + " " + lastName)
            frame = face_recognition.load_image_file(path + firstName + " " + lastName + ".jpg")
            Encodings.append(face_recognition.face_encodings(frame)[0])
            break

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


def FacialRecognitionProgram(Names, Encodings):
    video_capture = cv2.VideoCapture(0)
    
    #process_frame = True
    
    while(True):
        isReading, frame = video_capture.read() # read() outputs a value determining whether the operation ran succesfully and an image or frame
        small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25) #Resizing to 1/4 size for faster facial recognition processing
        small_frame = small_frame[:,:,::-1] #Converting from BGR(OpenCV uses this) to RGB(facial_recognition uses this)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        faces = zip(face_locations, face_encodings)
        for (top, right, bottom, left), current_face in faces:
            isPresent = face_recognition.compare_faces(Encodings, current_face, tolerance = 0.6)
            
            euclidean_distances = face_recognition.face_distance(Encodings, current_face) #Compare euclidean face distances of known faces to a face in the image
            best_match = numpy.argmin(euclidean_distances)
            
            if isPresent[best_match]:
                face_name = Names[best_match]
            else: face_name = "Unknown"
            
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
                
        if (cv2.waitKey(1) & 0xFF )==ord('q'): # Pressing q will exit the while loop and stop reading input from the camera
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def CameraProgram():
    video_capture = cv2.VideoCapture(0)
    handDetector = HandDetector(detectionCon=0.8, maxHands=1)
    path = sys.path[0] + "\Photos" + "\\"
    finished = Event()
    photoThread = Thread(target = takePhoto, args=(video_capture, finished))
    print("Camera Program #2 Loaded.\nTo take a photo, please make a peace symbol.\nTo exit the program, press 'q'\n")
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
                fingers1 = handDetector.fingersUp(hand1)
                if (fingers1 == [0, 1, 1, 0, 0]) and (not photoThread.is_alive()): #If user is holding a peace symbol
                    photoThread.start()
            if finished.is_set():
                finished.clear()
                cv2.imwrite(os.path.join(path,'image.jpg'), frame)
                photoThread = Thread(target = takePhoto, args=(video_capture,finished))
            
            cv2.imshow("Camera Program #2", image)
            cv2.waitKey(1) #1ms
    
        if (cv2.waitKey(1) & 0xFF )==ord('q'): # Pressing q will exit the while loop and stop reading input from the camera
            break
            
    video_capture.release()
    cv2.destroyAllWindows()
    
def takePhoto(vid, done):
    print("Taking Photo in 5 seconds.")
    for x in range(5,0,-1):
        print(x)
        sleep(1)
    print("Taking Photo.")
    isReading, img = vid.read()
    sleep(2)
    done.set()
    return

def UserInput(Names, Encodings):
    choice = input(
        "\nWhat would you like to do?\n1)Run Facial Recognition Program (Made Using OpenCV)\n2)Run Camera Program (Made Using CVZone)\n3)Train a new face\n")
    if choice == "1":
        FacialRecognitionProgram(Names, Encodings)
    elif choice == "2":
        CameraProgram()
    elif choice == "3":
        print("\nPlease have only 1 face in frame for picture.\n")
        trainFace(Names, Encodings)
    else:
        print("\nInvalid Input: Please enter an integer.\n")
    UserInput(Names, Encodings)


if __name__ == "__main__":
    main()