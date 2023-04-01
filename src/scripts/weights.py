import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import glob
import cv2

# Carregando as bases de imagens conhecidas e desconhecidas
known_images = glob.glob(r'src\assets\conhecidos\*')
unknown_dataset = glob.glob(r'src\assets\desconhecidos\*')

# Preprocessamento das imagens
def preprocess_image(image):
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
        image = cv2.resize(image, (32,32)) # Redimensiona a imagem para 32x32
        image = image.reshape((32, 32, 1)) # Adiciona um canal para a imagem
    return image

known_labels = [known_images[i].split('\\')[-1].split('.')[0] for i in range(len(known_images))]

unknown_images = [cv2.imread(file) for file in unknown_dataset]
unknown_images = [preprocess_image(image) for image in unknown_images if image is not None]

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
known_images = []
for file in glob.glob(r'src\assets\conhecidos\*'):
    image = cv2.imread(file)
    if image is not None:
        known_images.append(image)

known_images_array = np.array([preprocess_image(image) for image in known_images if image is not None])
model.fit(known_images_array, 
          np.arange(len(known_images)), 
          epochs=30)

weights_path = r'src\assets\weights'
# Salva os pesos do modelo treinado
model.save_weights(fr'{weights_path}\weights.h5')
