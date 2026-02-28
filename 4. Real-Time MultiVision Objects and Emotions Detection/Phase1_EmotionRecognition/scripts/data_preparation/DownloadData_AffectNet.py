# PURPOSE: This script unzips the AffectNet dataset into the appropriate directory for further processing.
print(
    "**************************************************************************\n"
    "The scripts import AffectNet dataset and unzip it into the 'data/raw_data/AffectNet' directory.\n"
    "**************************************************************************"
)

# 1. Setup Configuration
# 1.1. Load requested libraries
import zipfile 
import os
 
# 1.2. Define parameters
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (Phase1_EmotionRecognition)
PHASE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATASET_NAME = "AffectNet"

# 2. Unzip Dataset
zip_path = os.path.join(PHASE_ROOT, "data", "raw", f"{DATASET_NAME}.zip")

# 3. Extract files to the desired directory
destination_dir =  os.path.join(PHASE_ROOT, "data")
os.makedirs(destination_dir, exist_ok=True)

if os.path.exists(zip_path):
    # Unzip the dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)

    print(f"Message: {DATASET_NAME} dataset unzipped successfully at {destination_dir}")
else:
    print(f"Error: Zip file not found at {zip_path}")

# 4. Quick Check: Cardinality of extracted files
print("\nCardinality of extracted files:")

def count_files_in_directory(path):
    """Count files in a given directory."""
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

# List of directories to check
directories = [
    os.path.join(PHASE_ROOT, "data", "YOLO_format", "valid", "images"),
    os.path.join(PHASE_ROOT, "data", "YOLO_format", "train", "images"),
    os.path.join(PHASE_ROOT, "data", "YOLO_format", "test", "images")
]

# Affichage des résultats
sets = ["Train", "Test", "Valid"]

for dir_path, set_name in zip(directories, sets):
    file_count = count_files_in_directory(dir_path)
    print(f"Message: {set_name} --> {file_count} fichiers")
 
# END OF SCRIPT
