import cv2
import numpy as np

import os
import sys

from generic_methods import convertToGray

def prepareTrainingData(data_folder_path='training-data'):
    """Reads training images and returns two lists that relate a face
    with a label and two lists that relates a label/number with a person"""

    # Lists for relations img-number
    faces = []; labels = []
    # Lists for relation number-person
    numbers = []; names = []

    # Get folders names from data folder
    folder_names = os.listdir(data_folder_path)
    # Go through each directory and read images inside them
    for folder_name in folder_names:
        # Ignore anything that's not a folder and don't start with 's'
        if not os.path.isdir(folder_name) and not folder_name.startswith("s"):
            continue

        # Building dir path for later loading of imgs
        # Sample: subject_dir_path = "training-data/s1"
        subject_path = data_folder_path + '/' + folder_name

        # Getting label of current folder
        label = int(folder_name.replace("s", ""))
        numbers.append(label)

        # Validating that current folder has an owner (stated in name.txt)
        if not os.path.isfile(subject_path + '/name.txt'):
            print("Name for person in " + subject_path + " is required")
            exit(0)
        # Reading name from file inside current folder
        file = open(subject_path + '/name.txt')
        name = file.read()
        name = name.replace("\n", "")
        names.append(name)
        file.close()

        # Get names of imgs inside current folder
        subject_images_names = os.listdir(subject_path)
        # Add every cropped face image to list of faces
        for image_name in subject_images_names:
            # Ignore files that aren't images
            if not image_name.endswith('.jpg') and not image_name.endswith('.jpeg') and not image_name.endswith('.png'):
                continue

            # Build image path (smth like: image path = "training-data/s1/1.jpg")
            image_path = subject_path + '/' + image_name

            # Read image
            face = cv2.imread(image_path)
            face = convertToGray(face)

            # Add original pair
            faces.append(face)
            labels.append(label)

            # Adding more images for training
            # Get shortest shape
            if face.shape[0] < face.shape[0]:  # Height is shorter
                shortest = face.shape[0]
            else:  # Width is shorter
                shortest = face.shape[1]
            # Resize and add additional pairs
            for i in range(4):
                # Get factor to resize shortest size to 60, 90, 120, 150px
                factor = (60 + 30*i) / shortest
                new_face = cv2.resize(src=face, dsize=None, fx=factor, fy=factor)
                # Add additional pairs
                faces.append(new_face)
                labels.append(label)

    return faces, labels, numbers, names


def trainModel():
    """Generate face recognition model files using current training data"""
    print("Preparando datos de entrenamiento...")
    print()

    # Lists that relates a face with a label, and a label (number) with a name
    faces, labels, numbers, names = prepareTrainingData('training-data')

    print("Datos listos!")
    print()

    # Results of training data preparation
    print("Total de caras: ", len(faces))
    print("Total de etiquetas: ", len(labels))
    print("Relaciones:")
    for i in range(len(numbers)):
        print(str(numbers[i]) + " - " + names[i])
    print()

    # Create face recognizer and train it
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    # Delete previous model data
    if os.path.isfile('model/model.yml'):
        os.remove('model/model.yml')
    if os.path.isfile('model/profiles.txt'):
        os.remove('model/profiles.txt')

    # Save trained model
    face_recognizer.save('model/model.yml')

    # Save face recognition profiles
    file = open('model/profiles.txt', "w")
    for i in range(len(numbers)):
        file.write(str(numbers[i]) + "-" + names[i] + "\n")
    file.close()

    # Update model status
    file = open('model/status.txt', "w")
    file.write("Updated")
    file.close()