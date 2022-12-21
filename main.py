import sys
import cv2
import face_recognition
import pickle

def main():
    input = UserInput()
    if input == "1": FacialDetection()
    
def FacialDetection():
    #insert code here
    print("Hello World!") 

def UserInput():
    choice = input("\nWhat would you like to do?\n1)Run the Facial Detection Program\n2)Train a new face\n")
    if choice.isnumeric(): return choice
    else: print("\nInvalid Input: Please enter an integer.\n");UserInput()
    
if __name__ == "__main__":
    main()