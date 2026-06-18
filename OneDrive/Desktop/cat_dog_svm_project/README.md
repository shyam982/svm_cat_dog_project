# SVM Cats vs Dogs

This repository contains a simple SVM-based image classifier for the Kaggle Dogs vs Cats dataset.

Files:
- `cat_dogs.py` — main script to load images, train a model, evaluate and plot results.

Notes:
- The script will look for a `train/` folder containing the Kaggle images named like `cat.0.jpg`, `dog.0.jpg`.
- For convenience, if `train/` is missing the script now falls back to a synthetic demo dataset and a fast demo classifier.

Usage:
```bash
# Install dependencies (example):
python -m pip install -r requirements.txt

# Run the script
python cat_dogs.py
```

To run a full training on the real dataset, download and extract Kaggle's `train.zip` next to this script so the `train/` folder is present.

License: MIT
