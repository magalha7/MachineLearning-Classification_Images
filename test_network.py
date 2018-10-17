# -*- coding: utf-8 -*-

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

#construir os argumentos e o parse dos argumentos para serem passados pelo terminal
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

#carrega a imagem
image = cv2.imread(args["image"])
orig = image.copy()

#pre processamento da imagem para classificação
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#carrega a rede neural convolucional treinada
print("[INFO] Carregando a Rede...")
model = load_model(args["model"])

#classifica de acordo com a imagem de entrada
(cats, dogs) = model.predict(image)[0]

#Constroi o label
label = "Dog" if dogs > cats else "Gato"
proba = dogs if dogs > cats else cats
label = "{}: {:.2f}%".format(label, proba * 100)

#Cor do label da imagem
output = imutils.resize(orig, width=500)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_DUPLEX,
	1.0, (255, 255, 255), 2)

#MOstra o resultado de cada imagem
cv2.imshow("Resultado:", output)
cv2.waitKey(0)