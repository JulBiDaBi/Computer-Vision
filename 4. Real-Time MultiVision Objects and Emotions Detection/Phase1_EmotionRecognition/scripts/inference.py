import os
from ultralytics import YOLO

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (Phase1_EmotionRecognition)
PHASE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Load model
path_model = os.path.join(PHASE_ROOT, 'models', 'yolov8m_emotions_affectnet.pt')

if not os.path.exists(path_model):
    print(f"Error: Model not found at {path_model}")
    # Fallback to yolov8m.pt if custom model not found for demonstration
    print("Falling back to yolov8m.pt")
    path_model = 'yolov8m.pt'

model = YOLO(path_model)

# Predict on images
images_path = os.path.join(PHASE_ROOT, 'data', 'check_model_images')

if os.path.exists(images_path):
    image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('.jpeg', '.jpg', '.png'))]

    if not image_files:
        print(f"No images found in {images_path}")
    else:
        results = model.predict(source=image_files, save=True, conf=0.5, project=os.path.join(PHASE_ROOT, 'results'), name='inference')

        # Print results
        for result in results:
            print(f"Results for {result.path}:")
            # In a script, display(result.plot()) might not work as in a notebook
            # result.show() # This would open a window, might not be suitable for all environments
else:
    print(f"Images path not found: {images_path}")
