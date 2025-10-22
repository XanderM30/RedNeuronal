import firebase_admin
from firebase_admin import credentials, firestore, storage
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os

# -----------------------------
# 1️⃣ Inicializar Firebase
# -----------------------------
cred = credentials.Certificate("firebase_config/serviceAccountKey.json")

# ⚠️ Agregamos configuración de storageBucket al inicializar
firebase_admin.initialize_app(cred, {
    'storageBucket': 'neuromedx-77c11.appspot.com'  # 👈 Usa tu bucket real de Firebase
})

db = firestore.client()
bucket = storage.bucket()

# -----------------------------
# 2️⃣ Extraer enfermedades y síntomas
# -----------------------------
enfermedades_docs = db.collection("enfermedades").stream()
datos = {"enfermedades": []}

for doc in enfermedades_docs:
    enfermedad = doc.to_dict()
    datos["enfermedades"].append({
        "nombre": enfermedad["nombre"],
        "sintomas": enfermedad.get("sintomas", [])
    })

# -----------------------------
# 3️⃣ Procesar datos
# -----------------------------
enfermedades_nombres = [e["nombre"] for e in datos["enfermedades"]]

# Todos los síntomas únicos
sintomas = []
for e in datos["enfermedades"]:
    for s in e["sintomas"]:
        if s not in sintomas:
            sintomas.append(s)

# Entradas y salidas
X = []
y = []

for e in datos["enfermedades"]:
    vector = [1 if s in e["sintomas"] else 0 for s in sintomas]
    X.append(vector)
    salida = [1 if e["nombre"] == nombre else 0 for nombre in enfermedades_nombres]
    y.append(salida)

X = np.array(X)
y = np.array(y)

# -----------------------------
# 4️⃣ Definir y entrenar la red
# -----------------------------
model = Sequential([
    Dense(32, input_dim=len(sintomas), activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(enfermedades_nombres), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("⏳ Entrenando la red neuronal...")
model.fit(X, y, epochs=200, verbose=1)
print("✅ Entrenamiento completado")

# -----------------------------
# 5️⃣ Guardar modelo en local
# -----------------------------
model.save("modelo_enfermedades.h5")
print("✅ Modelo guardado como modelo_enfermedades.h5")

# -----------------------------
# 6️⃣ Convertir y subir a Firebase Storage
# -----------------------------
print("🔄 Convirtiendo modelo a TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modelo_enfermedades.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Modelo convertido a .tflite")

blob = bucket.blob("modelos/modelo_enfermedades.tflite")
blob.upload_from_filename("modelo_enfermedades.tflite")
print("☁️ Modelo subido a Firebase Storage en: modelos/modelo_enfermedades.tflite")

# -----------------------------
# 7️⃣ Función de predicción local (opcional)
# -----------------------------
def predecir_enfermedad(sintomas_paciente):
    if not os.path.exists("modelo_enfermedades.h5"):
        print("❌ No se encontró el modelo. Entrena primero la red.")
        return
    model = load_model("modelo_enfermedades.h5")
    vector = [1 if s in sintomas_paciente else 0 for s in sintomas]
    pred = model.predict(np.array([vector]))
    enfermedad_predicha = enfermedades_nombres[np.argmax(pred)]
    print("🧠 Enfermedad probable:", enfermedad_predicha)

# Ejemplo:
# predecir_enfermedad(["fiebre", "tos"])
