import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#from pgmpy.models import TreeAugmentedNaiveBayes
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1) Chargement des images (dataset plat)
# ----------------------------------------------------------------------------
def load_dataset_images_flat(root_dir: str):
    """
    Charge toutes les images depuis un dossier sans sous-dossiers.
    
    Chaque image est convertie en niveaux de gris pour le traitement SIFT.
    
    Args:
        root_dir: chemin du dossier contenant les images
    
    Returns:
        images: liste d'images (numpy arrays)
        filenames: liste des noms de fichiers correspondants
    """
    images = []
    filenames = []
    for fname in os.listdir(root_dir):
        path = os.path.join(root_dir, fname)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Conversion en niveaux de gris
        if img is None:
            continue
        images.append(img)
        filenames.append(fname)
    return images, filenames

# ----------------------------------------------------------------------------
# 2) Extraction SIFT
# ----------------------------------------------------------------------------
def extract_sift_descriptors(img, dense=True, step_size=8, patch_size=16):
    """
    Extrait les descripteurs SIFT d'une image.
    
    Args:
        img: image en niveaux de gris
        dense: si True, on utilise SIFT dense (points régulièrement espacés)
        step_size: espacement entre les points clés pour SIFT dense
        patch_size: taille du patch pour chaque point clé dense
    
    Returns:
        des: tableau numpy des descripteurs SIFT (N x 128)
    """
    sift = cv2.SIFT_create()
    if dense:
        h, w = img.shape
        # Génération de points clés régulièrement espacés
        kps = [
            cv2.KeyPoint(float(x), float(y), patch_size)
            for y in range(patch_size//2, h-patch_size//2, step_size)
            for x in range(patch_size//2, w-patch_size//2, step_size)
        ]
        _, des = sift.compute(img, kps)  # Calcul des descripteurs pour ces points
    else:
        _, des = sift.detectAndCompute(img, None)  # SIFT classique
    if des is None:
        return np.zeros((0,128), dtype=np.float32)  # Cas où aucun point clé n'est trouvé
    return des

# ----------------------------------------------------------------------------
# 3) Construction BoVW (Bag of Visual Words)
# ----------------------------------------------------------------------------
def build_vocabulary(all_descriptors, k=256):
    """
    Construit le vocabulaire BoVW en appliquant KMeans sur tous les descripteurs.
    
    Args:
        all_descriptors: liste de tous les descripteurs SIFT des images
        k: nombre de clusters / "visual words"
    
    Returns:
        kmeans: modèle KMeans entraîné
    """
    print("Concaténation des descripteurs...")
    stacked = np.vstack([des for des in all_descriptors if len(des) > 0])  # Mise en pile de tous les descripteurs
    print("Total descriptors:", stacked.shape)
    print("Entraînement MiniBatchKMeans pour BoVW...")
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, verbose=1)  # MiniBatch pour accélérer
    kmeans.fit(stacked)
    return kmeans

def hist_from_descriptors(des, kmeans):
    """
    Transforme les descripteurs d'une image en histogramme BoVW.
    
    Args:
        des: descripteurs SIFT d'une image
        kmeans: modèle KMeans du vocabulaire BoVW
    
    Returns:
        hist: histogramme normalisé (vecteur de taille k)
    """
    if des is None or len(des) == 0:
        return np.zeros(kmeans.n_clusters, dtype=np.float32)
    words = kmeans.predict(des)  # Assigne chaque descripteur à un cluster
    hist = np.bincount(words, minlength=kmeans.n_clusters).astype(np.float32)  # Compte des visual words
    hist /= (np.linalg.norm(hist) + 1e-9)  # Normalisation L2
    return hist

# ----------------------------------------------------------------------------
# 4) Clustering pour pseudo-labels
# ----------------------------------------------------------------------------
def cluster_histograms(histograms, n_clusters=5):
    """
    Effectue un clustering KMeans sur les histogrammes BoVW pour générer des pseudo-labels.
    
    Args:
        histograms: vecteurs BoVW des images
        n_clusters: nombre de clusters souhaité
    
    Returns:
        labels: pseudo-labels attribués par cluster
        kmeans: modèle KMeans entraîné
    """
    print(f"Clustering en {n_clusters} clusters pour pseudo-labels...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(histograms)
    return labels, kmeans

# ----------------------------------------------------------------------------
# 5) Visualisation des clusters
# ----------------------------------------------------------------------------
def visualize_clusters(histograms, labels, method='pca', title='Clusters BoVW'):
    """
    Visualise les clusters des images dans un plan 2D.
    
    Args:
        histograms: vecteurs BoVW
        labels: pseudo-labels
        method: 'pca' pour PCA ou 'tsne' pour t-SNE
        title: titre du graphique
    """
    print(f"Visualisation avec {method.upper()}...")
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    else:
        raise ValueError("method doit être 'pca' ou 'tsne'")
    
    H_reduced = reducer.fit_transform(histograms)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(H_reduced[:,0], H_reduced[:,1], c=labels, cmap='tab10', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Composante 1')
    plt.ylabel('Composante 2')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

# ----------------------------------------------------------------------------
# 6a) Entraînement SVM
# ----------------------------------------------------------------------------
def train_svm(histograms, labels):
    """
    Entraîne un classifieur SVM linéaire sur les histogrammes BoVW.
    
    Args:
        histograms: vecteurs BoVW
        labels: pseudo-labels ou labels réels
    
    Returns:
        svm: modèle SVM entraîné
    """
    print("Entraînement SVM...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(histograms, labels)
    return svm

# ----------------------------------------------------------------------------
# 6b) Entraînement TAN
# ----------------------------------------------------------------------------
def train_tan(histograms, labels, n_bins=6):
    """
    Entraîne un modèle TAN (Tree Augmented Naive Bayes) sur les histogrammes discrétisés.
    
    Args:
        histograms: vecteurs BoVW
        labels: pseudo-labels ou labels réels
        n_bins: nombre de bins pour la discrétisation
    
    Returns:
        tan_model: modèle TAN entraîné
        discretizer: KBinsDiscretizer utilisé
    """
    print("Discrétisation pour TAN...")
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    H_disc = discretizer.fit_transform(histograms).astype(int)
    


    # Construction DataFrame compatible avec pgmpy

    """
    Entraîne un modèle Tree-Augmented Naive Bayes (TAN) en utilisant 
    la méthodologie TreeSearch de pgmpy.
    """
    # 1. Construction DataFrame compatible avec pgmpy
    # NOTE : Assurez-vous que 'H_disc' et 'labels' sont bien définis ou passés en argument
    cols = [f"f{i}" for i in range(H_disc.shape[1])]
    df = pd.DataFrame(H_disc, columns=cols)
    df["class"] = labels.astype(int)
    
    # ------------------------------------------------------------------
    # 2. APPRENTISSAGE DE LA STRUCTURE (Remplacement de TreeAugmentedNaiveBayes())
    # ------------------------------------------------------------------
    print("Entraînement TAN...")

    # Initialisation de l'estimateur de structure
    # 'data=df' est nécessaire pour le TreeSearch
    est = TreeSearch(data=df, root_node="class")

    # Estimation de la structure TAN, en spécifiant le nœud de classe
    tan_model_dag = est.estimate(
        estimator_type="tan",
        class_node="class"
    )

    # 3. CRÉATION DU MODÈLE ET APPRENTISSAGE DES PARAMÈTRES
    
    # Création du modèle BayesianNetwork à partir de la structure apprise (les arêtes)
    tan_model = BayesianNetwork(tan_model_dag.edges())

    print("Apprentissage des paramètres...")
    
    # Entraînement des tables de probabilités (CPDs) du modèle sur les données
    # Le 'fit' ici apprend les paramètres pour la structure déjà définie.
    tan_model.fit(data=df)
    
    return tan_model, discretizer

# ----------------------------------------------------------------------------
# 7) Pipeline complet
# ----------------------------------------------------------------------------
def bovw_clustering_supervised(dataset_root="dataset",
                               k_bovw=128,
                               dense_sift=True,
                               step_size=8,
                               patch_size=16,
                               n_clusters=5,
                               use_tan=False,
                               n_bins=6,
                               visualize=True,
                               prefix="xray_bovw"):
    """
    Pipeline complet :
    - Chargement images
    - Extraction SIFT
    - Construction BoVW
    - Clustering pour pseudo-labels
    - Visualisation des clusters
    - Entraînement SVM ou TAN
    - Sauvegarde des artefacts
    
    Args:
        dataset_root: dossier contenant les images
        k_bovw: nombre de clusters BoVW
        dense_sift: True pour SIFT dense
        step_size: pas pour SIFT dense
        patch_size: taille patch pour SIFT dense
        n_clusters: nombre de clusters pour pseudo-labels
        use_tan: True pour TAN, False pour SVM
        n_bins: bins pour TAN
        visualize: True pour afficher les clusters
        prefix: préfixe pour fichiers sauvegardés
    """
    # 1) Chargement images
    images, filenames = load_dataset_images_flat(dataset_root)
    print(f"{len(images)} images chargées.")

    # 2) Extraction SIFT
    print("Extraction SIFT...")
    all_des = [extract_sift_descriptors(img, dense=dense_sift, step_size=step_size, patch_size=patch_size)
               for img in tqdm(images)]

    # 3) Construction BoVW
    kmeans_bovw = build_vocabulary(all_des, k=k_bovw)
    H = np.array([hist_from_descriptors(des, kmeans_bovw) for des in all_des], dtype=np.float32)

    # 4) Clustering pour pseudo-labels
    pseudo_labels, clustering_model = cluster_histograms(H, n_clusters=n_clusters)

    # 5) Visualisation
    if visualize:
        visualize_clusters(H, pseudo_labels, method='pca', title='Clusters BoVW PCA')

    # 6) Entraînement supervise
    if use_tan:
        model, discretizer = train_tan(H, pseudo_labels, n_bins=n_bins)
        dump(discretizer, f"{prefix}_discretizer.joblib")
    else:
        model = train_svm(H, pseudo_labels)

    # 7) Sauvegarde artefacts
    
    # 🔄 CORRECTION : Sauvegarde le Dictionnaire Visuel (kmeans_bovw) sous le nom attendu.
    # Si votre script de prédiction charge le clustering model avec ce nom, utilisez-le.
    dump(kmeans_bovw, f"{prefix}_clustering_model.joblib") 
    
    dump(model, f"{prefix}_{'tan' if use_tan else 'svm'}_model.joblib")
    dump(pseudo_labels, f"{prefix}_pseudo_labels.joblib")
    dump(filenames, f"{prefix}_filenames.joblib")
    
    # L'ancien clustering_model (pour les pseudo-labels) est moins important pour l'inférence
    # Si vous voulez quand même le sauvegarder, utilisez un nom distinct, par exemple :
    dump(clustering_model, f"{prefix}_pseudo_label_kmeans.joblib")
    
    print("✔ Pipeline terminé et artefacts sauvegardés.")

# ----------------------------------------------------------------------------
# 8) Exécution directe
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    bovw_clustering_supervised(
        dataset_root="dataset",
        k_bovw=128,
        dense_sift=True,
        step_size=8,
        patch_size=16,
        n_clusters=5,
        use_tan=False,   # True pour TAN, False pour SVM
        n_bins=6,
        visualize=True,
        prefix="xray_bovw"
    )
