#Importar las librerías necesarias
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import random

# Función para generar un color RGB único aleatorio
def generate_unique_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Función para dibujar las trayectorias históricas de los tracks
def render_tracks(image, track_memory, id, bbox, color_map):
    # Calcular el centro de la caja delimitadora
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2 - 60

    # Dibujar los puntos históricos de rastreo si existen
    if id in track_memory:
        for coord in track_memory[id]:
            cv2.circle(image, (coord[0], coord[1]), 5, color_map[id], -1)  # Dibujar un círculo lleno en cada punto

    # Actualizar la posición actual en la memoria de rastreo
    if id in track_memory:
        track_memory[id].append((center_x, center_y))
    else:
        track_memory[id] = [(center_x, center_y)]

# Función para detectar y rastrear personas en una imagen
def detect_and_track_individuals(image, yolo_model, sort_tracker, track_memory, color_map):
    detections = yolo_model(image, stream=True)

    # Procesar los resultados de detección
    for detection in detections:
        # Filtrar las cajas con confianza mayor a 0.7
        high_conf_indices = np.where(detection.boxes.conf.cpu().numpy() > 0.7)[0]
        bounding_boxes = detection.boxes.xyxy.cpu().numpy()[high_conf_indices].astype(int)
        
        # Actualizar los tracks con el algoritmo SORT
        updated_tracks = sort_tracker.update(bounding_boxes)
        updated_tracks = updated_tracks.astype(int)

        # Dibujar los tracks y las cajas de detección
        for x1, y1, x2, y2, track_id in updated_tracks:
            # Asignar un color único si es un nuevo track
            if track_id not in color_map:
                color_map[track_id] = generate_unique_color()

            # Llamar a la función para dibujar las trayectorias históricas
            render_tracks(image, track_memory, track_id, [x1, y1, x2, y2], color_map)
            # Dibujar texto con el ID de la persona
            cv2.putText(image, f"Persona {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 2, color_map[track_id], 2)
            # Dibujar el bounding box alrededor de la persona
            cv2.rectangle(image, (x1, y1), (x2, y2), color_map[track_id], 2)

    return image

def main():
    # Crear un objeto de captura de video desde un archivo de video
    video_capture = cv2.VideoCapture("/Users/donato/py proyects/Benito/videotest_t2.mp4")
    # Inicializar el modelo YOLO para la detección de objetos
    yolo_detector = YOLO("Benito/yolov8m.pt")
    # Inicializar el tracker SORT para el seguimiento de objetos
    sort_algorithm = Sort()
    # Diccionario para almacenar el historial de rastreo de cada track
    track_memory = {}
    # Diccionario para asignar colores únicos a cada track
    color_map = {}

    # Bucle principal de procesamiento de video
    while video_capture.isOpened():
        # Leer un frame del video
        ret, frame = video_capture.read()
        # Si no se puede leer el frame, salir del bucle
        if not ret:
            break

        # Detectar y rastrear personas en el frame actual
        frame = detect_and_track_individuals(frame, yolo_detector, sort_algorithm, track_memory, color_map)
        # Mostrar el frame procesado en una ventana
        cv2.imshow("Video Frame", frame)

        # Esperar 1 ms y salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberar los recursos de captura de video
    video_capture.release()
    # Cerrar todas las ventanas de OpenCV
    cv2.destroyAllWindows()

# Punto de entrada del script
if __name__ == "__main__":
    main()
