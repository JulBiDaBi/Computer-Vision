import os
from ultralytics import YOLO

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (Phase1_EmotionRecognition)
PHASE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Charger le modèle YOLOv8 pré-entraîné
model = YOLO("yolov8m.pt")

# Charger la configuration YAML
config = os.path.join(PHASE_ROOT, "data", "YOLO_format", "data_AffectNet.yaml")

# Entraînement du modèle
results = model.train(
    data=config,            # Fichier YAML
    epochs=100,             # Nombre d'époques
    imgsz=640,              # Taille des images
    batch=16,               # Taille du batch
    device="cpu",           # "cpu" ou "0" pour GPU
    lr0=0.001,              # Taux d'apprentissage initial
    optimizer="Adam",       # Optimiseur
    augment=True,           # Augmentation des données
    project=os.path.join(PHASE_ROOT, "results"),  # Dossier projet
    name="yolov8m_emotions_v1"     # Nom de l'expérience
)

# Évaluation automatique sur le set de validation
metrics = model.val()

# Test sur le set test
model.val(split="test")

# Export du modèle au format ONNX
model.export(format="onnx")

# Sauvegarde du modèle entraîné
model_path = os.path.join(PHASE_ROOT, "models", "yolov8m_emotions_affectnet.pt")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)

print(f"Modèle sauvegardé sous: {model_path}")
