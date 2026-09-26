# Face Authentication SaaS Portal

An enterprise-ready biometric authentication gateway featuring deep-learning face anti-spoofing (liveness detection), active blink challenge validation, and two-factor PIN security. The project is built using a modern, client-server decoupled architecture designed to run seamlessly in the cloud (AWS, GCP, Heroku, Docker) without requiring physical camera attachments on the host machine.

---

## Key Features

- **Client-Side Camera Capture**: HTML5 camera integration captures and transmits frames securely over stateless HTTP JSON APIs, ensuring compatibility with headless cloud servers.
- **Deep-Learning Liveness Ensemble**: Employs state-of-the-art models (including a CNN-ViT hybrid) to evaluate frames against presentation attacks (print, replay, mask).
- **Centroid Identity Store**: Combines 15 high-quality face registration crops spanning different illumination and facial angles into a unified biometric profile.
- **Active Blink Challenges**: Real-time Eye Aspect Ratio (EAR) tracking enforces a random challenge to ensure the presenter is live.
- **Two-Factor PIN Security**: Protects accounts with an optional secondary passcode layer, encrypted using cryptographic storage standards.
- **Security Policy Configurator**: Admin dashboard allows real-time configuration of rate-limits, lockout durations, risk thresholds, and active liveness criteria.
- **System File Integrity**: On-startup tamper-detection engine verifies source code, model configurations, and weight hashes against a secure manifest.

---

## Directory Structure

```
MajorProj/
├── checkpoints/          # Production-ready deep learning weights
│   ├── attention_cnn/    # Optimized Attention CNN model weight
│   ├── cdcn/             # Central Difference Convolution model weight
│   ├── cnn_vit/          # CNN-ViT Hybrid model weight (Active Default)
│   ├── ensemble/         # Bagging Ensemble model weight
│   └── frequency_cnn/    # Frequency CNN model weight
├── config/               # System Configurations
│   ├── config_auth.yaml  # Active security policies and model configuration
│   └── integrity_manifest.json # Cryptographic file hashes
├── data/                 # Data Storage
│   └── auth/             # Encrypted user identity stores, credentials, and logs
│       ├── attempts.csv  # Decrypted-on-demand authentication attempt log
│       ├── auth.key      # Identity encryption key
│       ├── pins.enc      # Cryptographically sealed 2FA PINs
│       ├── pins.key      # PIN encryption key
│       └── users.enc     # Cryptographically sealed biometric identity profiles
├── demo_web_cams/        # Utility launching scripts
│   └── Web_Frontend.bat  # Automated server startup script
├── results/              # Saved model evaluation diagrams
│   ├── accuracy_comparison.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── src/                  # Core PyTorch and Security Implementations
│   ├── auth/             # Identity storage, Rate limiting, and Verification managers
│   ├── datasets/         # Data transformations and validation augmentations
│   ├── liveness/         # Active blink challenge and eye tracking engine
│   ├── models/           # Deep learning model architecture definition scripts
│   ├── preprocessing/    # RetinaFace crop and alignment module
│   └── security/         # Startup file integrity verification class
├── static/               # CSS and theme assets
│   └── style.css         # UI design system
├── templates/            # HTML views (Login, Registration, Admin Panels)
├── generate_manifest.py  # Utility to update integrity manifest
├── requirements.txt      # List of dependencies
└── web_app.py            # Primary Flask server and controller endpoints
```

---

## Getting Started

### 1. Prerequisites
Make sure you have Python 3.9+ installed.

### 2. Setup and Installation

Create a virtual environment and install the required dependencies:
```bash
# Create and activate environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Unix/macOS

# Install package dependencies
pip install -r requirements.txt
```

### 3. Running the Server

Start the Flask application directly:
```bash
python web_app.py
```
Or use the provided batch script under `demo_web_cams/`:
```bash
demo_web_cams\Web_Frontend.bat
```

Once running, navigate to `http://localhost:5000` in your web browser.

---

## Administration

### Security Policy Real-time Customization
Log in to the Admin Dashboard (default: `admin`/`12345678`) to:
- Monitor login attempts, success/failure rates, and security logs in real-time.
- View evaluation metrics (Confusion Matrix, ROC curves, model accuracy comparisons).
- Enable/disable 2FA, lockout parameters, IP rate-limits, and challenge timeouts.
- Export audit logs or registered biometric identities.
