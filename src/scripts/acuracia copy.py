import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import cv2
import numpy as np
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, precision_recall_fscore_support
inicio = datetime.datetime.now()
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
predicted_labels = []
for i, unknown_image in enumerate(unknown_images):
    similarities = []
    for j, known_image in enumerate(known_images):
        # Predição da similaridade entre as duas imagens
        similarity = model.predict(np.array([unknown_image, known_image]))
        similarities.append(similarity[0][j])
    max_similarity = max(similarities)
    predicted_label = np.argmax(similarities)
    predicted_labels.append(predicted_label)
    print("Imagem %s: classificada como %s com %.2f%% similaridade" % (unknown_dataset[i].split("/")[-1].split(".")[0], 
                                                                        known_dataset[predicted_label].split("/")[-1].split(".")[0], 
                                                                        max_similarity*100))

# Calculando métricas de avaliação
precision, recall, thresholds = precision_recall_curve(unknown_labels, predicted_labels)
f1 = f1_score(unknown_labels, predicted_labels)
precision100 = precision_score(unknown_labels, predicted_labels, average='binary', pos_label=1)
recall100 = recall_score(unknown_labels, predicted_labels, average='binary', pos_label=1)

print("F1 score: %.2f" % f1)
print("Precision for class 1: %.2f" % precision100)
print("Recall for class 1: %.2f" % recall100)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Plotando a curva precision-recall
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plotando a matriz de confusão
cm = confusion_matrix(unknown_labels, predicted_labels)
plt.imshow(cm, cmap='binary')
plt.xticks([0,1], ['Known', 'Unknown'])
plt.yticks([0,1], ['Known', 'Unknown'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()
fim = datetime.datetime.now()
print('Tempo de execução:', fim - inicio)