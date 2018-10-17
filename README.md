Antes de executar os códigos do tp, instalar as seguintes bibliotecas que são muito usadas em Deep-Learning.

1)OpenCV
$ sudo apt-get install python-opencv

2)Imutils
$ pip install imutils

3) Dlib
$ pip install numpy
$ pip install scipy
$ pip install dlib

4) Scikit-learn
$ pip install -U scikit-learn

5) Scikit-image
$ pip install scikit-image

6) TensorFlow
$pip install tensorflow

7) Keras
$pip install keras

8) Mxnet
$ pip install mxnet

###################################################################################################################################################################

PRIMEIRO PASSO:

Executar o script para a rede ser criada e treinada além de definir por fim o modelo de dados que será salvo.

$ python train_network.py --dataset img_training --model "nome do seu modelo de saida".model

SEGUNDO PASSO:

Executar o script para testar a rede após ela ter sido treinada.

$ python test_network.py --model "nome do seu modelo de saida".model --image img_tests/"nome da imagem".jpg

###################################################################################################################################################################
OBS: O primeiro passo demora devido ao tamanho da base para treinamento