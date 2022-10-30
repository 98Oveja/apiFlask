#!/usr/bin/env python
# Este archivo usa el encoding: utf-8
# Keras
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
# Flask
from flask import Flask
from flask import render_template
from flask import Response

import mediapipe as mp
import time
import numpy as np
import cv2
import os

app = Flask(__name__)
mp_holistic = mp.solutions.holistic #modelo de mp
mp_drawing = mp.solutions.drawing_utils #importando utilidades
# Path del modelo preentrenado
# MODEL_PATH = 'models/modelNumbers.h5'
MODEL_PATH = 'models/modelFrases.h5'
# Cargamos el modelo preentrenado
model = load_model(MODEL_PATH)

# variables para deteccion
abc = ['a','b','c','h']
phrases = ['por favor','feliz','mucho gusto','perdóname']
# numbers = ['1','2','3','4','5','6','7','8','9','10']
# actions = np.concatenate((abc, phrases))
actions = np.array(['por favor','feliz','mucho gusto','perdoname','hola','adios','gracias','yo','ayuda'])
# actions = np.array(['1','2','3','4','5','6','7','8','9','10'])

cap = cv2.VideoCapture(0)


# generar deteccion mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # CONVERSIÓN DE COLOR BGR 2 RGB
    image.flags.writeable = False                  # La imagen ya no se puede escribir, por eso es false
    results = model.process(image)                 # realizar prediction
    image.flags.writeable = True                   # ahora se puede escribir en la img
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # conversion de color RGB 2 BGR
    return image, results

def draw_formateado_landmarks(image, results):
    # dibujar conexiones de cara
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1), 
            mp_drawing.DrawingSpec(color=(255,51,51), thickness=1, circle_radius=1)
            ) 
    # dibujar conexiones de poses
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            ) 
    # dibujar conexiones de mano izquierda
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            ) 
    # dibuajr conexiones de mano derecha
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            ) 
# Extraer los keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# detección
def detection():
    secuencia =[]
    sentencia = []
    predicciones = []
    threshold = 0.3
    traduction = ''

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # leer feed
            ret, frame = cap.read()

            # hacer la detección
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # marcas
            draw_formateado_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            secuencia.append(keypoints)
            secuencia = secuencia[-30:] #ultimos 30 puntos clave

            if len(secuencia) == 30:
                resultado = model.predict(np.expand_dims(secuencia, axis=0))[0]
                print(actions[np.argmax(resultado)])
                predicciones.append(np.argmax(resultado))

            # #3. Viz logic
                if np.unique(predicciones[-10:])[0]==np.argmax(resultado): 
                    if resultado[np.argmax(resultado)] > threshold: 
                        
                        if len(sentencia) > 0: 
                            if actions[np.argmax(resultado)] != sentencia[-1]:
                                sentencia.append(actions[np.argmax(resultado)])
                                traduction = actions[np.argmax(resultado)]
                        else:
                            sentencia.append(actions[np.argmax(resultado)])
                            traduction = actions[np.argmax(resultado)]

                if len(sentencia) > 5:
                    sentencia = sentencia[-5:]
                # Viz probabilities
                #image = prob_viz(resultado, actions, image, colors)
                cv2.rectangle(image, (340,640), (940, 680), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(traduction), (500,670), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            (flag, encodedImage) = cv2.imencode(".jpg", image)
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+
                    bytearray(encodedImage)+b'\r\n')

# @app.route("/")
# def index():
#     return render_template("prueba.html")

@app.route("/video_feed")
def video_feed():
    return Response(detection(),
        mimetype= "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)

cap.release()