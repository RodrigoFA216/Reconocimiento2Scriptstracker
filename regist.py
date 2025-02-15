import cv2
import mediapipe as mp
import numpy as np
import json
import os

def capture_image_from_camera():
    """Captura una imagen desde la cámara."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error al capturar la imagen desde la cámara.")
        return None
    return frame

def load_image_from_file(path):
    """Carga una imagen desde un archivo."""
    image = cv2.imread(path)
    if image is None:
        print("Error al cargar la imagen. Verifica la ruta.")
    return image

def get_user_choice():
    """Solicita al usuario elegir la fuente de la imagen."""
    choice = input("Elija la fuente de la imagen (1: Cámara, 2: Archivo): ")
    return choice.strip()

def triangle_area(p1, p2, p3):
    """
    Calcula el área de un triángulo dados sus vértices p1, p2 y p3
    utilizando la fórmula del área de Gauss (o fórmula del determinante).
    Cada punto es una tupla (x, y).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0
    return area

def extract_face_invariants(image):
    """
    Procesa la imagen para detectar el rostro y extraer los invariantes geométricos.
    Se usan los landmarks indicados para formar una malla con centro en el landmark 4.
    Se calculan 6 áreas (de triángulos formados por el centro y cada par consecutivo
    de puntos periféricos) y se normalizan para obtener invariantes relativos.
    """
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            print("No se detectó ningún rostro en la imagen.")
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape

        # Convertir los landmarks normalizados a coordenadas en píxeles.
        landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

        # Verificar que tengamos suficientes landmarks.
        if len(landmarks) < 468:
            print("No se detectaron suficientes landmarks faciales.")
            return None

        # Definir los índices de los landmarks a utilizar.
        indices = {
            "comisura_ojoi": 130,   # Comisura ojo izquierdo
            "lagrimal_i": 243,      # Lagrimal izquierdo
            "centro_nariz": 4,      # Centro de la nariz (se usará como centro)
            "lagrimal_d": 463,      # Lagrimal derecho
            "comisura_ojod": 359,   # Comisura ojo derecho
            "comisura_labial_i": 76,  # Comisura izquierda del labio
            "comisura_labial_d": 306  # Comisura derecha del labio
        }

        try:
            center = landmarks[indices["centro_nariz"]]
            # Lista de puntos periféricos en orden (se asume un orden que recorra la región de interés)
            peripheral = [
                landmarks[indices["comisura_ojoi"]],
                landmarks[indices["lagrimal_i"]],
                landmarks[indices["lagrimal_d"]],
                landmarks[indices["comisura_ojod"]],
                landmarks[indices["comisura_labial_d"]],
                landmarks[indices["comisura_labial_i"]]
            ]
        except IndexError:
            print("Error al extraer algunos landmarks.")
            return None

        # Calcular las 6 áreas de los triángulos formados por:
        # (center, peripheral[i], peripheral[(i+1)%6])
        areas = []
        for i in range(len(peripheral)):
            p1 = center
            p2 = peripheral[i]
            p3 = peripheral[(i+1) % len(peripheral)]
            area = triangle_area(p1, p2, p3)
            areas.append(area)

        total_area = sum(areas)
        if total_area == 0:
            print("El área total es cero; no se pudo calcular invariantes.")
            return None

        # Calcular invariantes relativos: cociente de cada área sobre el área total.
        invariants = {}
        for i, area in enumerate(areas):
            invariants[f"area_invariante_{i+1}"] = area / total_area

        return invariants

def save_invariants_to_json(invariants, label, json_file="invariants.json"):
    """
    Guarda (o actualiza) el archivo JSON con el nuevo registro de invariantes.
    Si el archivo ya existe, se cargan los datos y se añade el nuevo registro.
    """
    data = {}
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    data[label] = invariants
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Datos guardados en {json_file}")

def main():
    choice = get_user_choice()
    if choice == "1":
        image = capture_image_from_camera()
    elif choice == "2":
        path = input("Ingrese la ruta del archivo de imagen: ").strip()
        image = load_image_from_file(path)
    else:
        print("Opción no válida.")
        return

    if image is None:
        print("No se pudo obtener la imagen.")
        return

    invariants = extract_face_invariants(image)
    if invariants is None:
        print("No se pudieron extraer invariantes del rostro.")
        return

    label = input("Ingrese la etiqueta del registro: ").strip()
    save_invariants_to_json(invariants, label)

if __name__ == "__main__":
    main()
