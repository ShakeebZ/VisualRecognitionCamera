import sys
import cv2
import face_recognition
import pickle
import os


def main():
    names, encodings = initialTraining()
    input = UserInput()
    if input == "1":
        FacialDetection()
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


def FacialDetection():
    video_capture = cv2.VideoCapture(0)
    
    while(True):
        isReading, image = video_capture.read() # read() outputs a value determining whether the operation ran succesfully and an image
        cv2.imshow("Camera Feed", image) #imshow takes in a string parameter to name the feed and an image or frame
        
        #NEED TO FIX THIS, camera won't close when q is pressed
        
        if cv2.waitKey(1) & 0xFF ==('q'): # Pressing q will exit the while loop and stop reading input from the camera
            break
    
    video_capture.release()
    cv2.destroyAllWindows()


def UserInput():
    choice = input(
        "\nWhat would you like to do?\n1)Run the Facial Detection Program\n2)Train a new face\n")
    # Add Option for gestures to take photos later
    if choice.isnumeric():
        return choice
    else:
        print("\nInvalid Input: Please enter an integer.\n")
        UserInput()


if __name__ == "__main__":
    main()
