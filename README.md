# interpretable-X_ray Image-classification

Interpretable X-Ray Container Image Classification for Inspection Support
This project implements an interpretable computer vision pipeline for classifying X-ray scanner images from cargo container inspections.
The system is designed to support risk-based inspection prioritization by identifying visual patterns associated with potential non-compliance or concealed goods, while maintaining transparency and auditability.

Problem Statement
Cargo container inspection agencies face a fundamental challenge:
Extremely high inspection volumes
Limited human inspection capacity
Critical need for accountability and explainability
Deep learning models can achieve high accuracy but often act as black boxes, making them unsuitable for regulated or high-impact contexts where inspection decisions must be explainable and reviewable.
This project addresses the problem of detecting and classifying risk indicators in X-ray container images using interpretable visual representations, enabling AI-assisted inspection rather than automated enforcement.

Solution Overview
The system follows a classical, interpretable computer vision approach:
Local feature extraction using SIFT descriptors from X-ray images
Bag of Visual Words (BoVW) representation via clustering
Risk classification using:
Support Vector Machines (SVM)
Tree-Augmented Naive Bayes (TAN) for probabilistic reasoning
Inference pipeline that outputs human-interpretable risk scores or labels
This design favors traceability and explainability over opaque end-to-end deep learning.

Key Capabilities Demonstrated
Interpretable computer vision for X-ray imagery
Bag of Visual Words (BoVW) modeling
Probabilistic graphical models (Tree-Augmented Naive Bayes)
Classical ML applied to security inspection data
Modular separation of training, inference, and model loading
Responsible AI design for inspection support systems

Repository Structure

├── train_bovw_svm_tan.py          # Feature extraction, clustering, and model training

├── detect_and_classify_compliance.py     # Image classification and risk logic

├── inference.py                  # End-to-end inference pipeline

├── model_loader.py               # Model loading and abstraction layer

├── main.py                       # Orchestration / entry point

├── README.md

└── .gitignore

Large datasets, trained models, and raw X-ray images are intentionally excluded.


Key Code Components

Feature Extraction & Training
train_bovw_svm_tan.py implements:
SIFT feature extraction from X-ray images
Visual vocabulary creation via clustering
Training of SVM and TAN classifiers
Evaluation under imbalanced classification settings
This approach enables inspection analysts to reason about which visual patterns contribute to risk predictions.

Classification & Inference
detect_and_classify_compliance.py and inference.py define:
Consistent preprocessing between training and inference
Image-to-risk score transformation
Separation of model logic from orchestration code

Model Abstraction
model_loader.py isolates model loading logic, enabling:
API integration
Batch processing
Future deployment without refactoring core algorithms

Evaluation Philosophy
Evaluation prioritizes operational relevance over raw accuracy:
ROC AUC and PR AUC for rare event detection
Emphasis on false-negative reduction (missed risks)
Preference for interpretable error analysis
Metrics are used to support inspection prioritization, not automated decision-making.

Responsible AI Considerations
Designed strictly for decision support, not autonomous inspection
Interpretable visual features enable human verification
No sensitive or identifiable data included
No enforcement actions are automated
Large artifacts are excluded to prevent misuse
This aligns with ethical requirements in security, humanitarian, and regulatory environments.

Limitations & Future Work
Classical feature extraction may underperform deep CNNs on very large datasets
Performance may vary across scanner types and resolutions
Future extensions could include:
Fairness and bias analysis
Hybrid CNN + interpretable layer approaches
Integration with inspection case management systems

How to Run (Example):

detect_and_classify_compliance.py 

python train_bovw_svm_tan.py

python main.py

This repository focuses on methodology and structure rather than full production deployment.

Disclaimer

This project is provided for research and portfolio demonstration purposes only.
It does not represent a production enforcement or surveillance system.
