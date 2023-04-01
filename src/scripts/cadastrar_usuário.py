import tkinter as tk
import cv2
import os

class Application:
    def __init__(self, window):
        self.window = window
        self.window.title("Captura de imagem")
        
        # Cria um Frame para conter o botão e o campo de entrada
        self.input_frame = tk.Frame(self.window, width=640, height=50)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Cria o botão e o campo de entrada dentro do Frame
        self.nome_label = tk.Label(self.input_frame, text="Digite o nome:")
        self.nome_entry = tk.Entry(self.input_frame)
        self.capturar_button = tk.Button(self.input_frame, text="Capturar", command=self.capturar_imagem)
        
        # Define a posição do botão e do campo de entrada
        self.nome_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.nome_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        self.capturar_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Inicia a captura da imagem
        self.capture = cv2.VideoCapture(0)
        self.display = tk.Label(self.window)
        self.display.pack()
        self.show_frame()

    def capturar_imagem(self):
        # Captura a imagem da câmera
        ret, frame = self.capture.read()
        nome = self.nome_entry.get()
        caminho_pasta = r'C:\Users\bruno\OneDrive\Área de Trabalho\FaceRecognition_v2_HOG\src\assets\conhecidos'
        
        # Salva a imagem na pasta especificada
        caminho_arquivo = os.path.join(caminho_pasta, nome + ".jpg")
        cv2.imwrite(caminho_arquivo, frame)
        
        # Exibe uma mensagem de confirmação
        print("Cadastro realizado com sucesso!")
        
        # Libera a câmera e fecha a aplicação
        self.capture.release()
        self.window.destroy()

    def show_frame(self):
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
        self.display.config(image=img)
        self.display.img = img
        self.window.after(10, self.show_frame)
    
if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    root.mainloop()