import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import cv2
import numpy as np

# Carregando as bases de imagens conhecidas e desconhecidas
known_dataset = glob.glob(r'src\assets\conhecidos\*')
unknown_dataset = glob.glob(r'src\assets\desconhecidos\*')

# Preprocessamento das imagens
def preprocess_image(image):
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
        image = cv2.resize(image, (32,32)) # Redimensiona a imagem para 32x32
        image = image.reshape((32, 32, 1)) # Adiciona um canal para a imagem
    return image

known_images = [cv2.imread(file) for file in known_dataset]
known_images = [preprocess_image(image) for image in known_images if image is not None]

unknown_images = [cv2.imread(file) for file in unknown_dataset]
unknown_images = [preprocess_image(image) for image in unknown_images if image is not None]

# Criação dos rótulos
known_labels = np.arange(len(known_images))
unknown_labels = np.ones(len(unknown_images))

# Criação do modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(known_images), activation='softmax'))

# Compilação do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
model.fit(np.concatenate((known_images, unknown_images), axis=0), 
          np.concatenate((known_labels, unknown_labels), axis=0), 
          epochs=10, 
          validation_split=0.2)

# Avaliação das imagens desconhecidas
for i, unknown_image in enumerate(unknown_images):
    for j, known_image in enumerate(known_images):
        # Predição da similaridade entre as duas imagens
        similarity = model.predict(np.array([unknown_image, known_image]))
        print("Imagem %s x Imagem %s: %.2f%% similaridade" % (known_dataset[j].split("/")[-1].split(".")[0], unknown_dataset[i].split("/")[-1].split(".")[0], similarity[0][j]*100))
