
import joblib
from ultralytics import YOLO

def load_models(prefix="camcis_bovw"):
    kmeans = joblib.load(f"{prefix}_kmeans.joblib")
    discretizer = joblib.load(f"{prefix}_discretizer.joblib")
    tan_model = joblib.load(f"{prefix}_tan_model.joblib")
    classnames = joblib.load(f"{prefix}_classnames.joblib")

    # YOLOv8 model
    yolo_model = YOLO("yolov8n.pt")

    return {
        "kmeans": kmeans,
        "discretizer": discretizer,
        "tan": tan_model,
        "classnames": classnames,
        "yolo": yolo_model
    }
