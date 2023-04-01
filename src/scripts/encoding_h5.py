import cv2
import numpy as np
import glob

# Carregar as imagens de treinamento
train_images = glob.glob(r'src\assets\conhecidos\*')

# Criar um detector de faces
face_detector = cv2.CascadeClassifier(r'haarcascades\haarcascade_frontalface_default.xml')

# Extrair os rostos das imagens de treinamento
faces = []
for img_path in train_images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_rects = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces_rects:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (32, 32))
        faces.append(face)

# Salvar os rostos como um arquivo numpy
npy_path = r'src\assets\npy'
np.save(fr'{npy_path}\conhecidos.npy', faces)