import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import time, glob, os

# Carregando as bases de imagens conhecidas
known_dataset = glob.glob(r'src\assets\conhecidos\*')

# Preprocessamento das imagens
def preprocess_image(image):
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converte a imagem para escala de cinza
        image = cv2.resize(image, (32,32)) # Redimensiona a imagem para 32x32
        image = image.reshape((32, 32, 1)) # Adiciona um canal para a imagem
    return image

known_images = [cv2.imread(file) for file in known_dataset]
known_images = [preprocess_image(image) for image in known_images if image is not None]

# Criação do modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='softmax', name='dense_1'))

# Carregando o modelo treinado
model.load_weights(r'src\assets\weights\weights.h5')

# Captura de imagens da webcam
cap = cv2.VideoCapture(0)

# Define o valor de sensibilidade da comparação
sensitivity = 0.9

# Captura de imagens da webcam
cap = cv2.VideoCapture(0)

# Define o valor de sensibilidade da comparação
sensitivity = 0.9

while True:
    ret, frame = cap.read()

    # Preprocessamento da imagem da webcam
    image = preprocess_image(frame)

    # Comparação da imagem com a base de rostos conhecidos
    if image is not None:
        similarities = []
        for i, known_image in enumerate(known_images):
            # Predição da similaridade entre as duas imagens
            similarity = model.predict(np.array([known_image, image]))
            similarities.append(similarity[0][0])

        # Obtém o índice da imagem com a maior similaridade
        max_index = np.argmax(similarities)
        
        # Verifica se a maior similaridade é maior que o valor de sensibilidade
        if similarities[max_index] >= sensitivity:
            # Obtém o nome da pessoa associada à imagem com a maior similaridade
            name = os.path.splitext(os.path.basename(known_dataset[max_index]))[0]
            accuracy = similarities[max_index]
            color = (0, 255, 0) if accuracy >= 0.9 else (0, 0, 255) # Verde se a acurácia for maior ou igual a 0.9, vermelho caso contrário
                
            # Detecção do rosto na imagem da webcam
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + r"haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Desenho do retângulo ao redor do rosto
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            frame = cv2.rectangle(frame, (0, frame.shape[0]-50), (frame.shape[1], frame.shape[0]), color, cv2.FILLED) # Desenha um retângulo na parte inferior da imagem para exibir o nome e a acurácia
            frame = cv2.putText(frame, f'{name} - {accuracy:.2f}', (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # Exibe o nome e a acurácia na imagem

    cv2.imshow('frame', frame)

    # Encerra o loop quando a tecla 'q' é pressionada
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
