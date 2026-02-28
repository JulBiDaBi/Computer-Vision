import os
import yaml

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (Phase1_EmotionRecognition)
PHASE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Construire le contenu YAML
data_yaml = {
    "path": os.path.join(PHASE_ROOT, "data", "YOLO_format"),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 8,
    
    "names": {
        0: "Anger",
        1: "Contempt",
        2: "Disgust",
        3: "Fear",
        4: "Happy",
        5: "Neutral",
        6: "Sad",
        7: "Surprise"
    }
}

# Sauvegarder dans un fichier
file_path = os.path.join(PHASE_ROOT, "data", "YOLO_format", "data_AffectNet.yaml")
with open(file_path, "w") as f:
    yaml.dump(data_yaml, f)
    
print(f"Fichier YAML sauvegardé à : {file_path}")
