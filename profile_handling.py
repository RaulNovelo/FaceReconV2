import cv2
import numpy as np
# import socket

import os
import sys
import shutil

from generic_methods import *

MODEL_PATH = 'model'

def getNumbers(profiles_file_path=MODEL_PATH + '/profiles.txt'):
    """Returns a list of current numbers founded in profiles file"""
    numbers = []
    if os.path.isfile(profiles_file_path):
        file = open(profiles_file_path, "r")
        for line in file:
            # Get only numbers from profiles file
            number = int(line[:1])
            numbers.append(number)
        file.close()
        return numbers
    else:
        return numbers


def getNames(profiles_file_path=MODEL_PATH + '/profiles.txt'):
    """Returns a list of current names founded in profiles file"""
    names = []
    if os.path.isfile(profiles_file_path):
        file = open(profiles_file_path, "r")
        for line in file:
            # Get only names from profiles file
            name = line[2:len(line) - 1]
            names.append(name)
        file.close()
        return names
    else:
        return names


def getFacesFromWebcam(cropped_faces_path='training-data/temp/valid-imgs'):
    """Uses webcam to detect, crop and save faces"""
    # PERFORMANCE PARAMETERS
    min_face_size = 50
    max_face_size = 250
    frame_period = 15

    # FOLDER VALIDATION
    # Validates that a brand new folder is available to storage cropped faces
    if os.path.isdir(cropped_faces_path):
        shutil.rmtree(cropped_faces_path)
        os.mkdir(cropped_faces_path)
    else:
        os.mkdir(cropped_faces_path)

    # LOADING RESOURCES
    # Loading face detector
    face_detector = cv2.CascadeClassifier('xml-files/haarcascades/haarcascade_frontalface_default.xml')
    # Loading video feed
    video = cv2.VideoCapture(0)

    # PREPARATION
    # Instructions
    print("Press 'q' once to start recollecting images. The person"
          "should look straight to the camera and make different"
          "expressions. The person must not tilt his/her face to the "
          "sides.\n"
          "Press 'q' again to quit data recollection")

    # Counter for number of cropped faces
    crops = 0
    # Counter for number of captured frame
    current_frame = 0
    # Flag used to turn on face cropping mode
    cropping_is_active = False

    # READING FRAMES
    while True:
        # Read video frame by frame
        value, frame = video.read()
        if value == 0: # Skip empty frame
            continue

        # DETECTING FACES AND DISPLAYING VIDEO
        # Convert frame to gray scale for better detection accuracy
        gray_frame = convertToGray(frame)

        # Detect faces in frame
        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(min_face_size, min_face_size),
            maxSize=(max_face_size, max_face_size)
        )

        # Draw faces over frame
        for (x, y, h, w) in faces:
            frame = drawRectangleText(frame, (x, y, w, h), GREEN, "", GREEN)

        # Display resulting frame
        cv2.imshow("Video feed", frame)

        # CROPPING FACES
        # If cropping mode is active, count frames and...
        if cropping_is_active:
            current_frame += 1
            # ...if there is only one face, crop and save it
            if len(faces) == 1 and current_frame % frame_period == 0:
                (x, y, h, w) = faces[0]
                cropped_face = gray_frame[y:y + w, x:x + h]

                # Build path to save img
                crops += 1
                cropped_img_path = cropped_faces_path + '/' + str(crops) + '.jpg'
                # Save img
                cv2.imwrite(cropped_img_path, cropped_face)

        # Press 'q' to start face cropping. Press again to terminate recognition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if cropping_is_active:
                break
            else:
                cropping_is_active = True
                current_frame = 0
                crops = 0

    # When everything is done, release resources (webcam or RPi stream)
    video.release()
    cv2.destroyAllWindows()


def addProfile(model_data_path='model', media_folder_path='training-data/temp'):
    # NAME VALIDATION
    # Get current profiles names
    names = getNames(model_data_path + '/profiles.txt')

    # Validation loop
    profile_name = ""
    is_valid = False
    while not is_valid:
        profile_name = input("Ingrese el nombre de la persona: ")
        if profile_name in names:
            print("Nombre invalido")
        else:
            print("Nombre valido")
            is_valid = True

    # PROCESSING MEDIA
    # Validates that there is a folder to store incoming media
    if not os.path.isdir(media_folder_path):
        os.mkdir(media_folder_path)
    # Validates that a brand new folder is available to storage cropped faces
    if os.path.isdir(media_folder_path + '/valid-imgs'):
        shutil.rmtree(media_folder_path + '/valid-imgs')
        os.mkdir(media_folder_path + '/valid-imgs')
    else:
        os.mkdir(media_folder_path + '/valid-imgs')

    # [Prompt to choose way get media (from gallery, video sample, webcam) goes here]
    getFacesFromWebcam()

    # SAVING PROFILE
    # Get number for new profile (smallest integer available)
    numbers = getNumbers(model_data_path + '/profiles.txt')
    smallest = 1
    for i in sorted(numbers):
        if smallest == i:
            smallest += 1
        else:
            break
    profile_number = smallest

    # Build path for profile
    new_profile_path = 'training-data/s' + str(profile_number)
    # Create folder for profile
    os.mkdir(new_profile_path)

    # Move collected faces to profile folder
    files = os.listdir(media_folder_path + '/valid-imgs')
    for file in files:
        shutil.move(media_folder_path + '/valid-imgs/' + file, new_profile_path)

    # Make new name.txt file for profile
    file = open(new_profile_path + '/name.txt', "w")
    file.write(profile_name)
    file.close()

    # FINISH
    # Refresh status of current recognition model ("Outdated" means it needs retraining)
    file = open(model_data_path + '/status.txt', "w")
    file.write("Outdated")
    file.close()

    # Clean incoming media folder (to save storage space)
    shutil.rmtree(media_folder_path)


def showCurrentProfiles(profiles_file_path=MODEL_PATH + '/profiles.txt'):
    """Prints the names and numbers of current active profiles for face recognition"""
    if os.path.isfile(profiles_file_path):
        # Read profiles from file
        print("PERFILES ACTUALES")
        file = open(profiles_file_path)
        for line in file:
            print(line, end="")
        file.close()
    else:
        print("No existe perfil alguno")
