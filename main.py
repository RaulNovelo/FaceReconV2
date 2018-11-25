import cv2
import numpy as np
# import socket

import os
import sys
import shutil

# from generic_methods import *

PRY_PATH = os.path.dirname(os.path.abspath(__file__))

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
        # [Code of option 1 goes here]
        pass
    elif op==2:
        print("Entrenando modelo...")
        # [Code of option 2 goes here]
        pass
    elif op==3:
        print("Accediendo a perfiles...")
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
        # [Code of option 6 goes here]
        pass
    elif op==7:
        print("Recolectando caras...")
        # [Code of option 7 goes here]
        pass
    else:
        print("Opción no válida")
    print()
        

if __name__ == "__main__":
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
        elif op == 2:
            print("Entrenando modelo...")
            print()
            trainModel()
            print()
        elif op == 3:
            print("Accediendo a perfiles...")
            showCurrentProfiles()
            print()
        elif op == 4:
            print("Agregando perfil...")
            addProfile()
            print()
        elif op == 5:
            exit(0)
        elif op == 6:
            print()
            getFacesFromWebcam()
            print()
        else:
            print("Opcion no valida")
            print()
    """
