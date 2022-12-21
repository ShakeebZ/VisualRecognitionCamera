import sys
import cv2
import face_recognition
import pickle
import os

def main():
    input = UserInput()
    if input == "1": FacialDetection()
    
def FacialDetection():
    video_capture = cv2.VideoCapture(0)
    path = sys.path[0] + "\TrainingFaces"
    MatthewMercer_image = face_recognition.load_image_file(path + "\MatthewMercer.jpg")
    print("Success!")
    
    #Different Attempts at Accessing Image locations, using the join function results in the "\TrainingFaces" path being treated as an absolute path
    
    #MatthewMercer_image = face_recognition.load_image_file(os.path.join(sys.path[0],"\TrainingFaces\MatthewMercer.jpg"))
    #__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    #MatthewMercer_image = face_recognition.load_image_file(open(os.path.join(__location__, "\TrainingFaces\MatthewMercer.jpg"), "r"))

def UserInput():
    choice = input("\nWhat would you like to do?\n1)Run the Facial Detection Program\n2)Train a new face\n")
    if choice.isnumeric(): return choice
    else: print("\nInvalid Input: Please enter an integer.\n");UserInput()
    
if __name__ == "__main__":
    main()