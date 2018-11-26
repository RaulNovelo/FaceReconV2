import cv2
import numpy as np
# import socket

import os
import sys
import shutil

from generic_methods import *
from recognition import *
from training import trainModel
from profile_handling import *

PRY_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = PRY_PATH + 'model'
TRAIN_PATH = PRY_PATH + 'training-data'
CASCADES_PATH = PRY_PATH + 'xml-files'

"""
def startStreamServer(): # Not being used anymore
    h, p = sys.argv[1].split(' ')[0], 8000
    print("server running on", sys.argv[1].split(' ')[0])
    VideoRecognizer(h, p)
"""

def showMenu():
    menu = """-- MENU (Opciones para el menú final) --
        [ 1 ] --- Iniciar recon facial
        [ 2 ] --- Entrenar modelo
        [ 3 ] --- Ver perfiles del modelo
        [ 4 ] --- Agregar perfiles
        [ 5 ] --- Remover perfiles
        [ 6 ] --- Salir
        -- MENU (Opciones temporales a testear) --
        [ 7 ] --- Usar webcam para conseguir caras"""

    op = 0
    while(op != 6): 
        print(menu)
        print()
        try:
            op = int(input("Ingrese una opcion o (CTRL + C para salir): "))
            print()
            selectOption(op)
        except ValueError:
            print("Solo se permiten números")
            print()
            pass


def selectOption(op):
    if op==1:
        print("Iniciando reconocimiento...")
        recognizer.startWebcamRecon()
    elif op==2:
        print("Entrenando modelo...")
        trainModel()
    elif op==3:
        print("Accediendo a perfiles...")
        showCurrentProfiles()
        # [Code of option 3 goes here]
        pass
    elif op==4:
        print("Agregando perfil...")
        # [Code of option 4 goes here]
        pass
    elif op==5:
        print("Removiendo perfil...")
        # [Code of option 5 goes here]
        pass
    elif op==6:
        print("Saliendo...")
        pass
    elif op==7:
        print("Recolectando caras...")
        getFacesFromWebcam()
        pass
    else:
        print("Opción no válida")
    print()
        

if __name__ == "__main__":

    recognizer = VideoRecognizer() # Initialize host and port here

    try:
        showMenu()
    except KeyboardInterrupt:
        print()
        pass

    """
        if op == 1:
            # Validates status of current face recognition model
            if os.path.isfile("model/status.txt"):
                file = open("model/status.txt", "r")
                status = file.read()
                status = status.replace("\n", "")
                if status == "Updated":
                    if os.path.isfile("model/model.yml"):
                        # Start facial recognition
                        print("Reconociendo...")
                        print()
                        startRecon()
                        # startStreamServer()
                        print()
                    else:
                        # There's no model. Train and then recon
                        print("Entrenando y reconociendo...")
                        print()
                        trainModel()
                        startRecon()
                        # startStreamServer()
                        print()
                else:
                    # Model is out to date. Train and then recon
                    print("Actualizando y reconociendo...")
                    print()
                    trainModel()
                    startRecon()
                    # startStreamServer()
                    print()
            else:
                print("No existe status")
                exit(0)
    """
