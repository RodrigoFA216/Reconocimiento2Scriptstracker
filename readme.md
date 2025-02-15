# Face Invariants Recognition & Tracking

Este proyecto consta de dos scripts en Python que trabajan en conjunto para registrar y reconocer invariantes geométricos del rostro. Los invariantes se utilizan para reconocer y etiquetar rostros en tiempo real en un video, manteniendo la etiqueta asignada mientras el rostro es rastreado.

## Contenido

- **`capture_invariants.py`**:  
  Permite capturar una imagen (desde cámara o archivo), extraer los invariantes geométricos del rostro usando MediaPipe Face Mesh y guardar estos invariantes en un archivo JSON (`invariants.json`).

- **`video_recognition.py`**:  
  Detecta rostros en tiempo real desde la cámara, extrae los invariantes de cada rostro y los compara con los registros en `invariants.json`. Una vez reconocido un rostro, se crea un tracker (usando el tracker CSRT de OpenCV) que mantiene la etiqueta asignada. Además, cada 5 segundos se vuelve a extraer el rostro para actualizar la etiqueta y el tracker, asegurando una mayor robustez en el reconocimiento.

## Funcionalidades

- **Captura y Registro de Invariantes:**  
  - Utiliza MediaPipe Face Mesh para extraer landmarks faciales.
  - Calcula 6 invariantes (áreas de triángulos formados por el centro y puntos periféricos) normalizados sobre el área total.
  - Guarda los invariantes en `invariants.json` con una etiqueta asignada por el usuario.

- **Reconocimiento y Tracking en Video:**  
  - Detección de rostros en tiempo real con MediaPipe Face Mesh.
  - Cálculo de invariantes y matching con la base de datos usando la distancia Euclidiana.
  - Implementación de trackers (CSRT de OpenCV) para mantener la etiqueta mientras el rostro sea detectado.
  - Actualización cada 5 segundos: se reevalúa el rostro, se extraen los invariantes y se actualiza la etiqueta y el tracker si corresponde.

## Dependencias

- Python 3.x
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [MediaPipe](https://mediapipe.dev/)
- [NumPy](https://numpy.org/)

Puedes instalar las dependencias usando `pip`:

```bash
    pip install opencv-python mediapipe numpy
```

## Uso

Registro de Invariantes:

Ejecuta el script capture_invariants.py para capturar una imagen y registrar los invariantes del rostro.

```python
python capture_invariants.py
```

Se te preguntará si deseas usar la cámara o cargar una imagen desde archivo.
Ingresa la etiqueta (nombre) para identificar el registro.
Los invariantes se guardarán en invariants.json.

## Reconocimiento y Tracking en Video:

Ejecuta el script video_recognition.py para iniciar el reconocimiento en tiempo real.

```python
python video_recognition.py
```

El script abrirá la cámara, detectará los rostros y los etiquetará.
Cada 5 segundos se actualizará la información para mantener la etiqueta o cambiarla en función del matching con la base de datos.

Salida:
    Presiona la tecla q para salir del modo video.

## Personalización

Umbral de Matching:
En ambos scripts puedes ajustar el valor del umbral (por ejemplo, threshold=0.02) para modificar la sensibilidad del reconocimiento.

## Método de Tracking:

Se utiliza el tracker CSRT de OpenCV, pero puedes cambiar a otro tracker según tus necesidades (por ejemplo, KCF, MOSSE, etc.).

Créditos

- MediaPipe:
    Por proporcionar una excelente solución para la detección de landmarks faciales.

- OpenCV:
    Por su robusta implementación de procesamiento de imágenes y algoritmos de tracking.
