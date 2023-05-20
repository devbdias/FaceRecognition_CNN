import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import cv2
import numpy as np
import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
predicted_probabilities = model.predict(np.concatenate((known_images, unknown_images), axis=0))
predicted_labels = np.argmax(predicted_probabilities, axis=1)
true_labels = np.concatenate((known_labels, unknown_labels), axis=0)

# Métricas adicionais
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
macro_precision = precision_score(true_labels, predicted_labels, average='macro')
macro_recall = recall_score(true_labels, predicted_labels, average='macro')
macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

# Relatório de classificação
print("Relatório de Classificação:")
print(classification_report(true_labels, predicted_labels))

# Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(true_labels, predicted_labels))

# Métricas de desempenho
print("Acurácia:", accuracy)
print("Precisão (Weighted):", precision)
print("Recall (Weighted):", recall)
print("F1-score (Weighted):", f1)
print("Precisão (Macro):", macro_precision)
print("Recall (Macro):", macro_recall)
print("F1-score (Macro):", macro_f1)
