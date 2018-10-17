# -*- coding: utf-8 -*-
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
	@staticmethod
	#width = largura imagem
	#height = altura da imagem
	#depth = o número de canais na imagem no caso RGB = 3
	#classes = numero total de classes que queremos reconhecer no caso 2
	
	def build(width, height, depth, classes):
		# inicializa o modelo 
		model = Sequential()
		inputShape = (height, width, depth)

		# se estiver utilizando "channels first", atualizar a forma de entrada
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		#Começa adicionar camadas ao modelo criado.
		#primeiro camadas setar CONV => RELU => POOL 
		#same preenchimento de zero
		model.add(Conv2D(20, (5, 5), padding="same",input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		#segunda camadas CONV => RELU => POOL 
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# setar camadas FC => RELU
		model.add(Flatten()) #nivelamento de camadas para um vetor
		model.add(Dense(500)) #numero de nós
		model.add(Activation("relu"))

		# classificador softmax 
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# retorna a arquitetura de rede de neuronios construida
		return model