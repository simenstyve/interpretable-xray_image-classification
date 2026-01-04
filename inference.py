# inference.py

import cv2
import numpy as np

from model_loader import load_models

MODELS = load_models()   # charge une seule fois


# ---- SIFT + BoVW ---------------------------------------------------

def extract_sift(img_gray):
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(img_gray, None)
    if des is None:
        return np.zeros((1, 128), dtype=np.float32)
    return des


def hist_from_descriptors(des, kmeans):
    if des is None or len(des) == 0:
        return np.zeros(kmeans.n_clusters, dtype=np.float32)
    words = kmeans.predict(des)
    hist = np.bincount(words, minlength=kmeans.n_clusters).astype(np.float32)
    hist /= (np.linalg.norm(hist) + 1e-9)
    return hist


# ---- Mapping SH codes minimal --------------------------------------

SH_MAPPING = {
    "vehicle_complete": "8703",
    "electronics": "8542",
    "weapons": "9302",
    "organic": "0801",
}


# ---- Fonction principale -------------------------------------------

def detect_and_classify_compliance(image_path: str):

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image introuvable")

    # YOLOv8 detection
    results = MODELS["yolo"](image_path, verbose=False)
    det_objs = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            det_objs.append((cls_id, conf))

    # SIFT + BoVW histogram
    des = extract_sift(img)
    hist = hist_from_descriptors(des, MODELS["kmeans"]).reshape(1, -1)

    # Discretisation
    H_disc = MODELS["discretizer"].transform(hist).astype(int)

    # TAN inference
    # → returns probabilities over classnames
    proba = MODELS["tan"].predict_probability(
        [{"class": None, **{f"f{i}": int(H_disc[0, i]) for i in range(H_disc.shape[1])}}]
    )[0]

    # Convert probability dict → array
    classnames = MODELS["classnames"]
    probs = np.array([proba[f"class_{i}"] for i in range(len(classnames))])
    predicted_idx = int(np.argmax(probs))
    predicted_class = classnames[predicted_idx]

    # SH code mapping minimal
    sh_code = SH_MAPPING.get(predicted_class, "0000")

    # Compliance score simple (démo)
    score = float(np.max(probs)) * 100.0

    return {
        "predicted_class": predicted_class,
        "classname_index": predicted_idx,
        "confidence": float(np.max(probs)),
        "sh_code": sh_code,
        "compliance_score": score,
        "yolo_detections": det_objs
    }
