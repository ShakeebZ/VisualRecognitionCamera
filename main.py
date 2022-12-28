import sys
import cv2
import face_recognition
import numpy
import os
from cvzone.FaceDetectionModule import FaceDetector


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
                isPresent = face_recognition.compare_faces(Encodings, current_face)
                
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
            cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED) #Filled in rectangle below face for name
            
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
    detector = FaceDetector()
    
    while True:
        isReading, frame = video_capture.read()
        frame = detector.findHands(frame)
        hand_is_present, bbox = detector.findPosition(frame)
        
        cv2.imshow("Camera Program #2", frame)
        cv2.waitKey(1)
    

def UserInput():
    choice = input(
        "\nWhat would you like to do?\n1)Run Camera Program #1 (Made Using OpenCV)\n2)Run Camera Program #2 (Made Using CVZone)\n3)Train a new face\n")
    # Add Option for gestures to take photos later
    if choice.isnumeric():
        return choice
    else:
        print("\nInvalid Input: Please enter an integer.\n")
        UserInput()


if __name__ == "__main__":
    main()
