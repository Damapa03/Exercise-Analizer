.
# AI Pose Analyzer

Este proyecto utiliza **YOLOv8-Pose**, **OpenCV** y **yt_dlp** para analizar la postura en ejercicios físicos en tiempo real o desde vídeos de YouTube.

## 🚀 Utilidad
El analizador permite monitorizar la técnica de ejercicios comunes para prevenir lesiones y asegurar un entrenamiento efectivo. Detecta automáticamente si el usuario está de frente o de lado y ajusta los cálculos biomecánicos en consecuencia.

## 🛠️ Requisitos
Es necesario tener instalado Python 3.8+ y las siguientes librerías:
```bash
pip install ultralytics opencv-python numpy yt-dlp
```
*Nota: La primera vez que se ejecute, el script descargará automáticamente el modelo YOLOv8-Pose (aprox. 6MB para la versión nano).*

## 📖 Cómo se utiliza
Puedes ejecutar el script desde la terminal con diferentes parámetros:

1. **Analizar un vídeo de YouTube (por defecto):**
   ```bash
   python pruebamodelos.py --url "https://www.youtube.com/watch?v=ejemplo"
   ```

2. **Usar la Webcam en tiempo real:**
   ```bash
   python pruebamodelos.py --source 0
   ```

3. **Cambiar el tipo de ejercicio:**
   ```bash
   python pruebamodelos.py --exercise deadlift
   ```
   *Opciones: `squat` (sentadilla), `deadlift` (peso muerto), `pushup` (flexiones).*

## 🧠 ¿Qué hace el sistema?
- **Detección de Orientación:** Identifica si estás de lado o de frente para usar los puntos clave (keypoints) correctos.
- **Contador de Repeticiones:** Basado en los ángulos de las articulaciones (rodilla, cadera o codo).
- **Feedback en Tiempo Real:** Muestra mensajes de corrección (ej. "Baja más la cadera" o "Espalda inclinada").
- **HUD Informativo:** Panel lateral con ángulos exactos, FPS y estado de la postura (OK, Aviso, Error).
- **Grabación:** Genera un archivo `output_ejercicio.mp4` con todo el análisis visual superpuesto.