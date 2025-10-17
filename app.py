from flask import Flask, request, jsonify
from flask_cors import CORS # 1. Importar la extensión CORS
from predict_script import predict_image # Importar la función predict_image del paso 1
import io

app = Flask(__name__)
CORS(app) # 2. Habilitar CORS para permitir peticiones desde el frontend (React)

@app.route('/classify', methods=['POST'])
def classify_image():
    # 1. Verificar si se subió un archivo
    if 'file' not in request.files:
        # Añadido encabezado CORS por defecto de Flask.
        return jsonify({"error": "No se encontró el archivo de imagen"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Archivo no seleccionado"}), 400

    if file:
        # 2. Leer los bytes del archivo
        image_bytes = io.BytesIO(file.read())
        
        try:
            # 3. Llamar a la función de predicción del modelo
            # Se pasa los bytes de la imagen a la función predict_image (del paso 1)
            prediction = predict_image(image_bytes) 
            
            # 4. Devolver la respuesta de la clasificación
            return jsonify({
                "status": "success",
                "classification": prediction,
                "message": f"Clasificación realizada: {prediction}"
            }), 200
            
        except Exception as e:
            # Manejar cualquier error durante el preprocesamiento o predicción
            return jsonify({"error": f"Error en la inferencia del modelo: {str(e)}"}), 500

if __name__ == '__main__':
    # Usar puerto 5000 para desarrollo local. Render usará una variable de entorno.
    app.run(host='0.0.0.0', port=5000)
