from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("best.pt")  # Asegúrate de tenerlo en el mismo directorio

# Ruta de la imagen a analizar
image_path = "prueba.jpg"

# Ejecutar inferencia
results = model.predict(source=image_path, save=True, conf=0.4)

# Mostrar imagen con anotaciones (opcional)
img = cv2.imread("runs/detect/predict/prueba.jpg")

cv2.imshow("Resultado", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
