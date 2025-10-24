import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import os
import json
from itertools import combinations
from nltk.stem.snowball import SpanishStemmer  # pip install nltk
stemmer = SpanishStemmer()

# -----------------------------
# 1️⃣ Inicializar Firebase
# -----------------------------
cred = credentials.Certificate("firebase_config/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# 2️⃣ Extraer enfermedades y síntomas desde Firestore
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
# 3️⃣ Normalización de síntomas
# -----------------------------
def normalize(text):
    accents = 'áéíóúü'
    replacements = 'aeiouu'
    for i in range(len(accents)):
        text = text.replace(accents[i], replacements[i])
    return ''.join(c for c in text.lower() if c.isalnum() or c.isspace())

def stem_list(lst):
    return [stemmer.stem(normalize(s)) for s in lst]

enfermedades_nombres = [e["nombre"] for e in datos["enfermedades"]]

sintomas = []
sintomas_por_enfermedad = {}
for e in datos["enfermedades"]:
    sintomas_norm = stem_list(e["sintomas"])
    sintomas_por_enfermedad[e["nombre"]] = sintomas_norm
    for s in sintomas_norm:
        if s not in sintomas:
            sintomas.append(s)

# -----------------------------
# 3.1️⃣ Comprobar cambios con red anterior
# -----------------------------
control_file = "control_red.json"
cambio_detectado = True

if os.path.exists(control_file):
    with open(control_file, "r", encoding="utf-8") as f:
        control_data = json.load(f)

    prev_sintomas = []
    prev_enfermedades = []

    for e in control_data:
        prev_sintomas.extend(stem_list(e.get("sintomas", [])))
        if "nombre" in e:
            prev_enfermedades.append(e["nombre"])
        elif isinstance(e, str):
            prev_enfermedades.append(e)

    if sorted(prev_sintomas) == sorted(sintomas) and sorted(prev_enfermedades) == sorted(enfermedades_nombres):
        cambio_detectado = False

if cambio_detectado:
    print("⚠️ Se detectaron cambios en síntomas o enfermedades. La red se volverá a entrenar.")
else:
    print("✅ No hay cambios en síntomas o enfermedades. Puedes omitir reentrenar si quieres.")

# Guardar control
with open(control_file, "w", encoding="utf-8") as f:
    enfermedades_dict_list = []
    for e in datos["enfermedades"]:
        enfermedades_dict_list.append({
            "nombre": e["nombre"],
            "sintomas": e.get("sintomas", []),
            "tips": e.get("tips", [])
        })
    json.dump(enfermedades_dict_list, f, ensure_ascii=False, indent=4)

# -----------------------------
# 4️⃣ Preparar datos para la red
# -----------------------------
X = []
y = []

for e in datos["enfermedades"]:
    sintomas_norm = sintomas_por_enfermedad[e["nombre"]]
    # Limitar combinaciones a máximo 3 síntomas
    for r in range(1, min(len(sintomas_norm)+1, 4)):
        for combo in combinations(sintomas_norm, r):
            vector = [1 if s in combo else 0 for s in sintomas]
            X.append(vector)
            salida = [1 if e["nombre"] == nombre else 0 for nombre in enfermedades_nombres]
            y.append(salida)

X = np.array(X)
y = np.array(y)

# -----------------------------
# 5️⃣ Definir y entrenar la red con Dropout
# -----------------------------
model = Sequential([
    Dense(64, input_dim=len(sintomas), activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(enfermedades_nombres), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if cambio_detectado:
    print("⏳ Entrenando la red neuronal...")
    model.fit(X, y, epochs=400, verbose=1)
    print("✅ Entrenamiento completado")
else:
    print("⚠️ La red no fue entrenada porque no hubo cambios.")

# Guardar modelo
model.save("modelo_enfermedades.h5")
print("✅ Modelo guardado como modelo_enfermedades.h5")

# -----------------------------
# 6️⃣ Función de predicción avanzada con top-3
# -----------------------------
def predict_symptoms(user_input, top_n=3, threshold=5):
    input_words = stem_list(user_input.split())
    input_vector = [0] * len(sintomas)
    for i, s in enumerate(sintomas):
        for w in input_words:
            if w in s or s in w:
                input_vector[i] = 1
                break

    input_vector = np.array([input_vector])
    predictions = model.predict(input_vector)[0]

    results = []
    for i, prob in enumerate(predictions):
        if prob*100 >= threshold:
            results.append({
                "enfermedad": enfermedades_nombres[i],
                "probabilidad": round(float(prob)*100, 1)
            })

    results = sorted(results, key=lambda x: x["probabilidad"], reverse=True)
    return results[:top_n]

# Ejemplo de uso
usuario = "diarrea vomito"
resultado = predict_symptoms(usuario)
for r in resultado:
    print(f"{r['enfermedad']}: {r['probabilidad']}%")
