# -*- coding: utf-8 -*-

#define o backend do matplotlib para que os valores possam ser salvos em segundo plano
import matplotlib
matplotlib.use("Agg")

#importação de pacotes necessarios
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from initialModel.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

#construir os argumentos e o parse dos argumentos para serem passados pelo terminal
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

#inicializa o número de épocas para treinar, inicia taxa de aprendixado, e o tamanho do lote
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# inicializa o data e labels
print("[INFO]: Carregando Imagens...")
data = []
labels = []

# pega o caminho das imagens e embaralha aleatoriamente
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# pré processamento das imagens
for imagePath in imagePaths:
	#carregar a imagem, pré-processá-la e armazená-la na lista de dados
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28)) #tamanho de 28x28 pixels uso da lenet
	image = img_to_array(image) 
	data.append(image)

	# extraia o rótulo de classe do caminho da imagem e atualiza o labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "dogs" else 0
	labels.append(label)

#scala de intensidade de pixel do rgb
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

'''
Particionar os dados em divisões de treinamento e teste usando 75%
os dados para treinamento e os 25% restantes para testes
'''
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

#converter os labels para vetores de inteiros
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# constroi o gerador de imagem para aumento de dados
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# inicializa o modelo
print("[INFO] Compilando o Modelo...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# traina a rede
print("[INFO] Treinando a Rede...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# salva o modelo no disco
print("[INFO] Serializando a Rede...")
model.save(args["model"])

#plota a perda e a precisão do treinamento
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Treinamento - Perda e Acertos em Dogs")
plt.xlabel("Epocas")
plt.ylabel("Perdas/Acertos")
plt.legend(loc="lower left")
plt.savefig(args["plot"])