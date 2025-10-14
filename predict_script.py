import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 1. Definición de la Red Neuronal
# Usaremos un modelo simple de dos capas convolucionales y tres capas fully-connected
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Capa Convolucional 1: 3 canales de entrada (RGB), 6 canales de salida, kernel 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # Capa Convolucional 2: 6 canales de entrada, 16 canales de salida, kernel 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Capas Fully-Connected (el tamaño de entrada dependerá del tamaño de la imagen original)
        # Suponiendo que las imágenes son preprocesadas a un tamaño específico (ej. 32x32)
        # El número 16*5*5 viene del cálculo: (16 canales * tamaño_final_del_feature_map)
        # Para un ejemplo simple (ej. CIFAR-10), esto podría ser 16*5*5 = 400
        # En tu caso, para una prueba simple, el tamaño *exacto* no importa, solo el flujo.
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        # Capa de salida: 10 clases (ej. si usáramos CIFAR-10, o 2 para tu radiografía: normal/anormal)
        self.fc3 = nn.Linear(84, 2) # Ajusta esto a 2 (Normal, Anormal) para tu prueba de radiografía
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Aplanar para las capas fully-connected
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Función de Inferencia
def predict_image(image_bytes, model_path='simple_cnn.pth'):
    # Clases que el modelo simula predecir
    classes = ['Normal', 'Anormal'] 
    
    # 2.1. Cargar el modelo
    model = SimpleNet()
    # Asume que ya tienes un modelo simple entrenado guardado
    # En un MVP real, entrenarías esto con un pequeño dataset.
    # Para la prueba, simularemos la carga:
    try:
        # Aquí deberías cargar el archivo .pth con torch.load(model_path)
        # Como es una prueba, puedes omitir el load y devolver una predicción simulada.
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.eval() # Poner el modelo en modo de evaluación
        pass # Simulación: simplemente pasamos

    except Exception as e:
        print(f"Error simulado al cargar el modelo: {e}. Usando predicción aleatoria.")

    # 2.2. Preprocesar la imagen
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        # Usar la normalización adecuada para el dataset de entrenamiento (ej. ImageNet)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_bytes).convert('RGB')
    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0) # Añadir dimensión de lote (batch dimension)
    
    # 2.3. Hacer la predicción (SIMULADA para el MVP)
    # En un escenario real:
    # with torch.no_grad():
    #     outputs = model(tensor)
    #     _, predicted = torch.max(outputs.data, 1)
    #     predicted_class_index = predicted.item()
    
    # SIMULACIÓN: Predicción aleatoria para demostrar el flujo de la API
    import random
    predicted_class_index = random.randint(0, 1) # 0 o 1
    
    return classes[predicted_class_index]
