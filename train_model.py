import firebase_admin
from firebase_admin import credentials, firestore, storage
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os
import json

# -----------------------------
# 1️⃣ Inicializar Firebase
# -----------------------------
cred = credentials.Certificate("firebase_config/serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    'storageBucket': 'neuromedx-77c11.appspot.com'
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

sintomas = []
for e in datos["enfermedades"]:
    for s in e["sintomas"]:
        if s not in sintomas:
            sintomas.append(s)

# -----------------------------
# 3.1️⃣ Comprobar cambios con red anterior
# -----------------------------
control_file = "control_red.json"
cambio_detectado = True  # Asumimos que hay cambios por defecto

if os.path.exists(control_file):
    with open(control_file, "r", encoding="utf-8") as f:
        control_data = json.load(f)
    prev_sintomas = control_data.get("sintomas", [])
    prev_enfermedades = control_data.get("enfermedades", [])
    # Comparamos
    if sorted(prev_sintomas) == sorted(sintomas) and sorted(prev_enfermedades) == sorted(enfermedades_nombres):
        cambio_detectado = False

if cambio_detectado:
    print("⚠️ Se detectaron cambios en síntomas o enfermedades. La red se volverá a entrenar.")
else:
    print("✅ No hay cambios en síntomas o enfermedades. Puedes omitir reentrenar si quieres.")

# Guardamos la nueva lista para la próxima comparación
with open(control_file, "w", encoding="utf-8") as f:
    json.dump({"sintomas": sintomas, "enfermedades": enfermedades_nombres}, f, ensure_ascii=False, indent=4)

# -----------------------------
# 4️⃣ Entradas y salidas
# -----------------------------
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
# 5️⃣ Definir y entrenar la red
# -----------------------------
model = Sequential([
    Dense(32, input_dim=len(sintomas), activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(enfermedades_nombres), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if cambio_detectado:
    print("⏳ Entrenando la red neuronal...")
    model.fit(X, y, epochs=200, verbose=1)
    print("✅ Entrenamiento completado")
else:
    print("⚠️ La red no fue entrenada porque no hubo cambios.")

# -----------------------------
# 6️⃣ Guardar modelo en local
# -----------------------------
model.save("modelo_enfermedades.h5")
print("✅ Modelo guardado como modelo_enfermedades.h5")

# -----------------------------
# 7️⃣ Convertir y subir a Firebase Storage
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
