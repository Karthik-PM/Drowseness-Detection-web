import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
model = load_model(r'C:\Users\dheer\Documents\DEEP LEARNING\apple vs orange\drowse.h5')
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    resize = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat >= 0.3:
        window_name = 'Image'
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = cv2.putText(img, 'sleep', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow(window_name, image)
    else:
        window_name = 'Image'
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = cv2.putText(img, 'awake', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow(window_name, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()