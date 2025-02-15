import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# Cargar invariantes registrados
if not os.path.exists("invariants.json"):
    print("No se encontró el archivo invariants.json")
    invariants_db = {}
else:
    with open("invariants.json", "r") as f:
        invariants_db = json.load(f)
    # invariants_db tiene la estructura:
    # {
    #    "Nombre": {
    #         "area_invariante_1": ...,
    #         "area_invariante_2": ...,
    #         ...
    #    },
    #    ...
    # }

# Índices de landmarks a utilizar
LANDMARK_INDICES = {
    "comisura_ojoi": 130,     
    "lagrimal_i": 243,        
    "centro_nariz": 4,        
    "lagrimal_d": 463,        
    "comisura_ojod": 359,     
    "comisura_labial_i": 76,  
    "comisura_labial_d": 306  
}

def triangle_area(p1, p2, p3):
    """Calcula el área de un triángulo dados sus vértices."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0
    return area

def compute_invariants_from_landmarks(landmarks, image_shape):
    """
    Extrae los invariantes geométricos a partir de los landmarks.
    Se calcula el área de 6 triángulos formados por el landmark central (índice 4)
    y cada par consecutivo de puntos periféricos.
    """
    h, w, _ = image_shape
    coords = [(lm.x * w, lm.y * h) for lm in landmarks]
    
    try:
        center = coords[LANDMARK_INDICES["centro_nariz"]]
        peripheral = [
            coords[LANDMARK_INDICES["comisura_ojoi"]],
            coords[LANDMARK_INDICES["lagrimal_i"]],
            coords[LANDMARK_INDICES["lagrimal_d"]],
            coords[LANDMARK_INDICES["comisura_ojod"]],
            coords[LANDMARK_INDICES["comisura_labial_d"]],
            coords[LANDMARK_INDICES["comisura_labial_i"]]
        ]
    except IndexError:
        return None

    areas = []
    for i in range(len(peripheral)):
        p1 = center
        p2 = peripheral[i]
        p3 = peripheral[(i+1) % len(peripheral)]
        areas.append(triangle_area(p1, p2, p3))
        
    total_area = sum(areas)
    if total_area == 0:
        return None

    invariants = {}
    for i, area in enumerate(areas):
        invariants[f"area_invariante_{i+1}"] = area / total_area
    return invariants

def compute_distance(inv1, inv2):
    """Calcula la distancia Euclidiana entre dos vectores de invariantes."""
    diff_sq = 0.0
    for key in inv1.keys():
        diff_sq += (inv1[key] - inv2.get(key, 0))**2
    return np.sqrt(diff_sq)

def match_invariants(face_invariants, invariants_db, threshold=0.02):
    """
    Compara los invariantes extraídos del rostro con la base de datos.
    Si la distancia Euclidiana es menor al umbral, se retorna el nombre;
    de lo contrario se retorna "desconocido".
    """
    best_match = "desconocido"
    best_distance = float("inf")
    for name, db_inv in invariants_db.items():
        distance = compute_distance(face_invariants, db_inv)
        if distance < best_distance and distance < threshold:
            best_distance = distance
            best_match = name
    return best_match

def get_bounding_box_from_landmarks(landmarks, image_shape):
    """Calcula la caja delimitadora a partir de todos los landmarks."""
    h, w, _ = image_shape
    coords = [(lm.x * w, lm.y * h) for lm in landmarks]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return x_min, y_min, x_max - x_min, y_max - y_min

# Clase para almacenar tracker y su etiqueta asociada
class FaceTracker:
    def __init__(self, tracker, label, bbox):
        self.tracker = tracker  # Tracker de OpenCV
        self.label = label      # Etiqueta actual (nombre o "desconocido")
        self.bbox = bbox        # Última posición conocida (x, y, w, h)
        self.lost_frames = 0    # Contador de frames en que falló el tracker

def initialize_tracker(frame, bbox, label):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    return FaceTracker(tracker, label, bbox)

def iou(bbox1, bbox2):
    """
    Calcula la Intersección sobre Unión (IoU) entre dos bounding boxes.
    Cada bbox es (x, y, w, h).
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+w1, x2+w2)
    yB = min(y1+h1, y2+h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = w1 * h1
    boxBArea = w2 * h2
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    trackers = {}  # Diccionario: key -> FaceTracker
    update_interval = 5  # segundos entre re-evaluaciones
    last_update_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Actualizar los trackers activos en cada frame
        remove_keys = []
        for key, face_tracker in trackers.items():
            ok, bbox = face_tracker.tracker.update(frame)
            if ok:
                x, y, w_box, h_box = map(int, bbox)
                face_tracker.bbox = (x, y, w_box, h_box)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                cv2.putText(frame, face_tracker.label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                face_tracker.lost_frames = 0
            else:
                face_tracker.lost_frames += 1
                if face_tracker.lost_frames > 10:
                    remove_keys.append(key)
        for key in remove_keys:
            trackers.pop(key)

        # Cada 5 segundos, actualizamos las detecciones y, si corresponde, la etiqueta y el tracker
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Obtener bbox de la detección actual
                    new_bbox = get_bounding_box_from_landmarks(face_landmarks.landmark, frame.shape)
                    # Extraer invariantes para la detección actual
                    invariants = compute_invariants_from_landmarks(face_landmarks.landmark, frame.shape)
                    if invariants is None:
                        continue
                    # Realizar matching para obtener la etiqueta
                    label = match_invariants(invariants, invariants_db, threshold=0.02)
                    
                    # Verificar si esta detección se asocia a un tracker existente (usando IoU)
                    associated = False
                    for key, face_tracker in trackers.items():
                        if iou(new_bbox, face_tracker.bbox) > 0.5:
                            # Se asocia: actualizamos etiqueta y re-inicializamos el tracker
                            new_tracker = cv2.TrackerCSRT_create()
                            new_tracker.init(frame, new_bbox)
                            face_tracker.tracker = new_tracker
                            face_tracker.label = label
                            face_tracker.bbox = new_bbox
                            associated = True
                            break
                    # Si no se asocia a ningún tracker, creamos uno nuevo
                    if not associated:
                        new_face_tracker = initialize_tracker(frame, new_bbox, label)
                        trackers[len(trackers)] = new_face_tracker

            last_update_time = current_time

        cv2.imshow("Video - Presione 'q' para salir", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
