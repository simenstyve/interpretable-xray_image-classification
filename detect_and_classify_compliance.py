"""
detect_and_classify_compliance.py

Version SVM :
 - YOLOv8 pour detection
 - SIFT + BoVW pour histogrammes
 - SVM (au lieu de TAN/pgmpy) pour classification
 - Combinaison score : alpha*yolo + beta*svm_proba
"""

import os
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from joblib import load
import pandas as pd

# ============================================================
# 1) Chargement artefacts (remplacement TAN par SVM)
# ============================================================
def load_bovw_svm_artifacts(prefix: str):
    """
    prefix: ex 'xray_bovw'
    fichiers attendus :
      prefix_clustering_model.joblib  -> KMeans/MiniBatchKMeans
      prefix_svm_model.joblib             -> SVM final
      prefix_pseudo_labels.joblib     -> pseudo-labels
      prefix_filenames.joblib         -> si besoin
    """

    kmeans = load(prefix + "_clustering_model.joblib")          # votre dictionnaire visuel
    svm_model = load(prefix + "_svm_model.joblib")                  # <-- SVM remplace TAN
    pseudo_labels = load(prefix + "_pseudo_labels.joblib")      # pseudo labels (info facultative)
    
    # vous pouvez charger ceci si utile pour debug
    try:
        filenames = load(prefix + "_filenames.joblib")
    except:
        filenames = None

    # classes = labels uniques triées
    class_names = sorted(list(set(pseudo_labels)))

    return kmeans, svm_model, class_names


# ============================================================
# 2) Préprocessing X-Ray
# ============================================================
def preprocess_xray_crop(crop: np.ndarray) -> Optional[np.ndarray]:
    if crop is None:
        return None
    if crop.dtype != np.uint8:
        img = crop.astype(np.float32)
        mx = img.max() if img.max() > 0 else 1.0
        img_u8 = (img / (mx + 1e-9) * 255).astype(np.uint8)
    else:
        img_u8 = crop

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(img_u8)
    bl = cv2.bilateralFilter(cl, d=7, sigmaColor=50, sigmaSpace=50)
    return bl.astype(np.float32) / 255.0


# ============================================================
# 3) SIFT extraction
# ============================================================
def extract_sift_descriptors(img: np.ndarray,
                             dense: bool = True,
                             step_size: int = 8,
                             patch_size: int = 16,
                             contrastThreshold: float = 0.01,
                             edgeThreshold: float = 5,
                             sigma: float = 1.2) -> Optional[np.ndarray]:

    if img is None:
        return None

    if img.dtype != np.uint8:
        img_u8 = (img * 255).astype(np.uint8)
    else:
        img_u8 = img

    sift = cv2.SIFT_create(contrastThreshold=contrastThreshold,
                           edgeThreshold=edgeThreshold,
                           sigma=sigma)

    if dense:
        h, w = img_u8.shape
        kps = []
        for y in range(patch_size//2, h - patch_size//2, step_size):
            for x in range(patch_size//2, w - patch_size//2, step_size):
                kps.append(cv2.KeyPoint(float(x), float(y), size=patch_size))
        if len(kps) == 0:
            return None
        _, des = sift.compute(img_u8, kps)
    else:
        _, des = sift.detectAndCompute(img_u8, None)

    return des


# ============================================================
# 4) Histogramme BoVW
# ============================================================
def hist_from_descriptors(descriptors: np.ndarray, kmeans) -> Optional[np.ndarray]:
    if descriptors is None or len(descriptors) == 0:
        return None

    words = kmeans.predict(descriptors)
    k = kmeans.n_clusters

    hist = np.bincount(words, minlength=k).astype(np.float32)
    hist = hist / (np.linalg.norm(hist) + 1e-9)
    return hist


# ============================================================
# 5) Prédiction SVM (remplacement TAN → SVM)
# ============================================================
def svm_predict_histogram(hist: np.ndarray,
                          svm_model,
                          class_names: List[str]) -> Tuple[Optional[str], float]:

    """
    hist : vecteur BoVW normalisé
    svm_model : SVM entraîné (avec probability=True si vous voulez les probabilités)
    Retourne (classe_préduite, probabilité)
    """

    if hist is None:
        return None, 0.0

    X = hist.reshape(1, -1)

    # ------------------------------
    # 🔄 **REMPLACEMENT TAN -> SVM**
    # ------------------------------
    pred_idx = svm_model.predict(X)[0]

    # Probabilité (si votre SVM a été entraîné avec probability=True)
    try:
        proba = svm_model.predict_proba(X)[0][pred_idx]
    except:
        proba = 0.5  # fallback si pas de probas

    cls_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    return cls_name, float(proba)


# ============================================================
# 6) YOLO wrapper
# ============================================================
def detect_yolo(image: np.ndarray, yolo_model: YOLO, conf: float = 0.25, iou: float = 0.45):
    if image.ndim == 2:
        inp = np.stack([image, image, image], axis=-1)
    else:
        inp = image

    results = yolo_model.predict(source=inp, conf=conf, iou=iou)[0]

    dets = []
    for box in results.boxes:
        xyxy = [float(x) for x in box.xyxy[0].tolist()]
        conf_score = float(box.conf[0].item())
        cls_id = int(box.cls[0].item()) if hasattr(box, 'cls') else None
        cls_name = yolo_model.names[cls_id] if (cls_id is not None and cls_id in yolo_model.names) else None

        dets.append({
            'xyxy': xyxy,
            'conf': conf_score,
            'cls_id': cls_id,
            'cls_name': cls_name
        })
    return dets


# ============================================================
# 7) Fonction principale
# ============================================================
def detect_and_classify_compliance(image_path: str,
                                   yolo_weights_path: str,
                                   bovw_prefix: str,
                                   hs_mapping: Dict[str, str],
                                   dense_sift: bool = True,
                                   yolo_conf: float = 0.25,
                                   yolo_iou: float = 0.45,
                                   bbox_padding: int = 8,
                                   alpha: float = 0.4,
                                   beta: float = 0.6) -> List[Dict]:

    # Charger artefacts
    kmeans, svm_model, class_names = load_bovw_svm_artifacts(bovw_prefix)
    yolo_model = YOLO(yolo_weights_path)

    img = cv2.imread(image_path)

    # Convert RGBA → RGB
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)    
        
    # Convert to grayscale for BoVW/SIFT
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # YOLO
    detections = detect_yolo(img, yolo_model, conf=yolo_conf, iou=yolo_iou)

    h_img, w_img = img_gray.shape
    final_results = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])

        # padding
        x1p = max(0, x1 - bbox_padding)
        y1p = max(0, y1 - bbox_padding)
        x2p = min(w_img - 1, x2 + bbox_padding)
        y2p = min(h_img - 1, y2 + bbox_padding)

        if x2p <= x1p or y2p <= y1p:
            continue

        crop = img_gray[y1p:y2p, x1p:x2p]
        crop_p = preprocess_xray_crop(crop)

        des = extract_sift_descriptors(crop_p, dense=dense_sift)
        hist = hist_from_descriptors(des, kmeans)

        # -----------------------------
        # 🔄 PRÉDICTION SVM au lieu de TAN
        # -----------------------------
        svm_cls, svm_prob = svm_predict_histogram(hist, svm_model, class_names)

        hs_code = hs_mapping.get(svm_cls, "0000")

        yolo_c = det.get("conf", 0.0) or 0.0
        svm_p = float(svm_prob or 0.0)

        compliance_score = float(alpha * yolo_c + beta * svm_p)

        final_results.append({
            "bbox": [int(x1p), int(y1p), int(x2p), int(y2p)],
            "class": svm_cls,
            "hs_code": hs_code,
            "yolo_conf": float(yolo_c),
            "svm_prob": float(svm_p),
            "compliance_score": compliance_score
        })

    return final_results


# ============================================================
# 8) Exemple d'utilisation
# ============================================================
if __name__ == "__main__":
    sample_image = "container_vehicle.png"
    yolo_weights = "yolov8n.pt"
    bovw_prefix = "xray_bovw"

    hs_mapping_example = {
        "vehicle_complete": "8703",
        "vehicle_parts": "8708",
        "electronics": "8517",
        "metal_objects": "7208",
        "unknown": "0000"
    }

    results = detect_and_classify_compliance(
        image_path=sample_image,
        yolo_weights_path=yolo_weights,
        bovw_prefix=bovw_prefix,
        hs_mapping=hs_mapping_example
    )

    print("Résultats SVM :")
    for r in results:
        print(r)
