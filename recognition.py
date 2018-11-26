import cv2
import numpy as np
# import socket

import os

from generic_methods import *

class VideoRecognizer(object):
    # def __init__(self, host, port):
    def __init__(self):
        # self.server_socket = socket.socket()
        # self.server_socket.bind((host, port))
        # self.server_socket.listen(0)
        # self.connection, self.client_address = self.server_socket.accept()
        # self.connection = self.connection.makefile('rb')
        # self.host_name = socket.gethostname()
        # self.host_ip = socket.gethostbyname(self.host_name)
        self.data = 1


    def loadSubjects(self):
        relations = {}
        file = open('model/profiles.txt', "r")
        for line in file:
            line = line.replace("\n", "")
            relations[int(line[0])] = line.replace(line[0] + "-", "")
        file.close()
        self.subjects = relations
        # return relations
        

    def loadModel(self):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read('model/model.yml')
        # return face_recognizer # self.recognizer = face_recognizer


    def performPrediction(self, face):
        """Recognizes the face of a person in the image and
        returns information about him/her"""
        # Recognize face
        # Note: predict() returns label=(int number, double confidence)
        prediction = self.face_recognizer.predict(face)

        # Search person who it's related to the number returned by predict()...
        if prediction[1] < 100:  # ...if confidence is small enough
            if prediction[0] in self.subjects:  # ... and if that number is registered in profiles.txt
                name = self.subjects[prediction[0]]
            else:
                name = "Not registered"
        else:
            name = "Unknown"  # ...otherwise, its an unknown person

        # Build text to be draw in the image (confidence is converted to percentage)
        confidence = 100 - prediction[1]
        recognition_info = name + " - " + format(confidence, ".2f") + "%"

        return recognition_info


    def startWebcamRecon(self):
        # DEFINING PARAMETERS (for best performance)
        min_face_size = 50
        max_face_size = 200

        # LOADING RESOURCES
        # subjects, model
        self.loadSubjects() # Relations number-person (smth like {1: "Fernando", 2: "Esteban", ...})
        self.loadModel()  # Trained model
        face_detector = cv2.CascadeClassifier('xml-files/lbpcascades/lbpcascade_frontalface.xml') # Cascade classifier
        video = cv2.VideoCapture(0) # Video stream

        # READING VIDEO
        while True:
            avaliability, frame = video.read()
            if avaliability == 0:  # Skip empty frame
                continue

            # Convert frame to gray scale
            gray_frame = convertToGray(frame)

            # Detecting faces in frame
            frontal_faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(min_face_size, min_face_size),
                maxSize=(max_face_size, max_face_size)
            )

            # PROCESSING EACH FACE IN FRAME
            for (x, y, h, w) in frontal_faces:
                cropped_face = gray_frame[y:y + w, x:x + h] # Crop face
                recognition_info = self.performPrediction(cropped_face) # Recognize face
                frame = drawRectangleText(frame, (x, y, h, w), GREEN, recognition_info, GREEN) # Draw rectangle and text

            # Display resulting frame
            cv2.imshow("Webcam video feed", frame)

            # Recognition will stop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Deteniendo reconocimiento...")
                print()
                break

        # When everything is done, release resources (webcam or RPi stream)
        video.release()
        cv2.destroyAllWindows()


    def startStreamRecon(self):
        # DEFAULT SIZES
        # 40-160 is a good range for RaspiCam detection up to 4 meters
        min_face_size = 45
        max_face_size = 155

        # LOAD RESOURCES
        # Load detectors
        frontal_face_detector = cv2.CascadeClassifier('xml-files/lbpcascades/lbpcascade_frontalface.xml')
        # stop_sign_detector = cv2.CascadeClassifier('xml-files/haarcascades/stop_sign.xml')
        # traffic_light_detector = cv2.CascadeClassifier('xml-files/haarcascades/traffic_light.xml')
        # lateral_face_detector = cv2.CascadeClassifier('xml-files/haarcascades/haarcascade_profileface.xml')
        self.loadSubjects() # Load subjects (for prediction)
        self.loadModel() # Load trained model

        try:
            print("Host: ", self.host_name + ' ' + self.host_ip)
            print("Connection from: ", self.client_address)
            print("Streaming...")
            print("Press 'q' to exit")

            # Need bytes here
            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]

                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    gray = convertToGray(frame)

                    frontal_faces = frontal_face_detector.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=8,
                        minSize=(min_face_size, min_face_size),
                        maxSize=(max_face_size, max_face_size)
                    )

                    # Draw a rectangle around the faces
                    for (x, y, w, h) in frontal_faces:
                        cropped_face = gray[y:y + w, x:x + h] # Crop face
                        recognition_info = self.performPrediction(cropped_face) # Recognize face
                        print(recognition_info)
                        frame = drawRectangleText(frame, (x, y, h, w), GREEN, recognition_info, GREEN) # Draw rectangle and text

                    # for (x, y, w, h) in stop_signs:
                    #    frame = drawRectangleText(frame, (x, y, h, w), RED, "Stop", RED
                    #    print('Stop sign detected')

                    # Debug face range rectangles
                    # frame = drawRectangleText(frame, (0, 0, max_face_size, max_face_size), BLUE, None)
                    # frame = drawRectangleText(frame, (0, 0, min_face_size, min_face_size), RED, None)

                    # Display the resulting frame
                    cv2.imshow('RPI video stream feed', frame)

                    # Press 'q' to stop face recognition
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.connection.close()
            self.server_socket.close()
