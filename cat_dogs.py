
# ══════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════
import os
import sys
import time
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
# CONFIGURATION  ← change these as needed
# ══════════════════════════════════════════════
DATASET_PATH  = "train"       # folder with cat.*.jpg / dog.*.jpg
IMG_SIZE      = (64, 64)      # resize resolution (bigger = slower but more accurate)
MAX_IMAGES    = 200           # total images to use (None = all ~25 000)
N_PCA         = 100           # PCA dimensions
SVM_C         = 10            # SVM penalty
SVM_GAMMA     = "scale"       # kernel coefficient
SVM_KERNEL    = "rbf"         # 'rbf' | 'linear' | 'poly'
TEST_SIZE     = 0.20          # 20 % held-out test set
RANDOM_STATE  = 42
MODEL_FILE    = "svm_model.pkl"
DO_GRID_SEARCH = False        # True = find best C/gamma (slow, ~30 min)
DEMO_MODE = False             # set True when falling back to synthetic data


# ══════════════════════════════════════════════
# STEP 1 — LOAD & PREPROCESS IMAGES
# ══════════════════════════════════════════════
def load_images(folder, img_size, max_images=None):
    """
    Read images from the Kaggle Dogs-vs-Cats train folder.
    Label: 0 = cat, 1 = dog
    Returns X  shape (N, H*W*3) float32  and  y shape (N,) int
    """
    if not os.path.isdir(folder):
        print(f"\n⚠️  Folder '{folder}' not found. Falling back to synthetic data for demo.")
        # Create a small synthetic dataset so the script can run without the Kaggle data.
        n = max_images or 200
        n = int(n)
        half = n // 2
        X = np.random.rand(n, img_size[0] * img_size[1] * 3).astype(np.float32)
        y = np.array([0] * half + [1] * (n - half))
        # shuffle
        perm = np.random.RandomState(RANDOM_STATE).permutation(n)
        X = X[perm]
        y = y[perm]
        print(f"\n  ✓ Generated synthetic dataset: {n} images  |  cats={np.sum(y==0)}  dogs={np.sum(y==1)}")
        global DEMO_MODE
        DEMO_MODE = True
        return X, y

    all_files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # balanced subsample
    if max_images:
        cats = [f for f in all_files if f.startswith("cat")][:max_images // 2]
        dogs = [f for f in all_files if f.startswith("dog")][:max_images // 2]
        all_files = cats + dogs

    X, y = [], []
    print(f"\n{'─'*55}")
    print(f"  STEP 1 › Loading {len(all_files)} images …")
    print(f"{'─'*55}")

    for fname in tqdm(all_files, unit="img", ncols=70):
        label = 0 if fname.startswith("cat") else 1
        path  = os.path.join(folder, fname)
        img   = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img.flatten().astype(np.float32) / 255.0)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"\n  ✓ Loaded {len(X)} images  |  cats={np.sum(y==0)}  dogs={np.sum(y==1)}")
    print(f"  ✓ Feature vector size = {X.shape[1]}  ({img_size[0]}×{img_size[1]}×3)")
    return X, y


# ══════════════════════════════════════════════
# STEP 2 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════
def split_data(X, y):
    print(f"\n{'─'*55}")
    print(f"  STEP 2 › Splitting data  (train 80% / test 20%)")
    print(f"{'─'*55}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  ✓ Train : {len(X_train)} samples")
    print(f"  ✓ Test  : {len(X_test)}  samples")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════
# STEP 3 — BUILD PIPELINE
#   StandardScaler → PCA → SVM
# ══════════════════════════════════════════════
def build_pipeline(n_pca, C, gamma, kernel):
    """
    Pipeline:
      1. StandardScaler  – zero-mean, unit-variance per pixel feature
      2. PCA             – reduce 12 288 dims → n_pca dims (speeds up SVM enormously)
      3. SVC             – RBF Support Vector Classifier
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, whiten=True, random_state=RANDOM_STATE)),
        ("svm",    SVC(C=C, gamma=gamma, kernel=kernel,
                       probability=True, random_state=RANDOM_STATE)),
    ])


# ══════════════════════════════════════════════
# STEP 4 — OPTIONAL GRID SEARCH
# ══════════════════════════════════════════════
def grid_search(X_train, y_train):
    print(f"\n{'─'*55}")
    print(f"  STEP 4 › Grid Search for best C & gamma  (this may take a while …)")
    print(f"{'─'*55}")
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=N_PCA, whiten=True, random_state=RANDOM_STATE)),
        ("svm",    SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.0001],
    }
    gs = GridSearchCV(base, param_grid, cv=3, scoring="accuracy",
                      n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)
    print(f"\n  ✓ Best params : {gs.best_params_}")
    print(f"  ✓ Best CV acc : {gs.best_score_*100:.2f}%")
    return gs.best_estimator_, gs.best_params_


# ══════════════════════════════════════════════
# STEP 5 — TRAIN
# ══════════════════════════════════════════════
def train_model(X_train, y_train):
    print(f"\n{'─'*55}")
    print(f"  STEP 5 › Training SVM")
    print(f"    kernel={SVM_KERNEL}  C={SVM_C}  gamma={SVM_GAMMA}  PCA={N_PCA}")
    print(f"{'─'*55}")

    if DO_GRID_SEARCH:
        model, best_params = grid_search(X_train, y_train)
    else:
        if DEMO_MODE:
            print("  ⚡ DEMO_MODE: using a fast DummyClassifier for quick demo runs")
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=min(10, N_PCA), whiten=True, random_state=RANDOM_STATE)),
                ("svm", DummyClassifier(strategy="most_frequent")),
            ])
        else:
            model = build_pipeline(N_PCA, SVM_C, SVM_GAMMA, SVM_KERNEL)

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  ✓ Training complete in {elapsed:.1f}s")
    joblib.dump(model, MODEL_FILE)
    print(f"  ✓ Model saved → {MODEL_FILE}")
    return model


# ══════════════════════════════════════════════
# STEP 6 — EVALUATE
# ══════════════════════════════════════════════
def evaluate_model(model, X_test, y_test):
    print(f"\n{'─'*55}")
    print(f"  STEP 6 › Evaluation")
    print(f"{'─'*55}")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]   # prob of being dog
    acc     = accuracy_score(y_test, y_pred)

    print(f"\n  ╔══════════════════════════════╗")
    print(f"  ║  Test Accuracy : {acc*100:6.2f} %   ║")
    print(f"  ╚══════════════════════════════╝")
    print()
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    return y_pred, y_proba, acc


# ══════════════════════════════════════════════
# STEP 7 — VISUALISATIONS
# ══════════════════════════════════════════════
def plot_all(model, X_test, y_test, y_pred, y_proba, acc):
    print(f"\n{'─'*55}")
    print(f"  STEP 7 › Generating visualisations …")
    print(f"{'─'*55}")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Task-03 | SVM Cats vs Dogs — Results", fontsize=16, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── A. Confusion Matrix ──────────────────
    ax_cm = fig.add_subplot(gs[0, 0])
    cm    = confusion_matrix(y_test, y_pred)
    disp  = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(f"Confusion Matrix\n(Accuracy {acc*100:.1f}%)", fontsize=11)

    # ── B. ROC Curve ────────────────────────
    ax_roc = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc      = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0,1],[0,1],"k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")

    # ── C. PCA Cumulative Variance ──────────
    ax_pca = fig.add_subplot(gs[0, 2])
    pca    = model.named_steps["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax_pca.plot(cumvar, color="darkorange", lw=2)
    ax_pca.axhline(90, color="red", ls="--", lw=1, label="90% variance")
    ax_pca.set_xlabel("PCA Components")
    ax_pca.set_ylabel("Cumulative Variance (%)")
    ax_pca.set_title("PCA Explained Variance")
    ax_pca.legend()

    # ── D. Confidence Score Distribution ────
    ax_dist = fig.add_subplot(gs[1, 0])
    cat_probs = y_proba[y_test == 0]
    dog_probs = y_proba[y_test == 1]
    ax_dist.hist(cat_probs, bins=30, alpha=0.6, color="royalblue",  label="True Cat")
    ax_dist.hist(dog_probs, bins=30, alpha=0.6, color="tomato",     label="True Dog")
    ax_dist.axvline(0.5, color="black", ls="--", lw=1, label="Decision boundary")
    ax_dist.set_xlabel("Predicted Probability (Dog)")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Confidence Distribution")
    ax_dist.legend(fontsize=8)

    # ── E. Class Balance ────────────────────
    ax_bar = fig.add_subplot(gs[1, 1])
    labels  = ["Cat (0)", "Dog (1)"]
    counts  = [np.sum(y_test==0), np.sum(y_test==1)]
    colors  = ["royalblue", "tomato"]
    bars    = ax_bar.bar(labels, counts, color=colors, edgecolor="white", width=0.5)
    for bar, cnt in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(cnt), ha="center", va="bottom", fontweight="bold")
    ax_bar.set_title("Test Set Class Distribution")
    ax_bar.set_ylabel("Number of Images")

    # ── F. Per-class Accuracy ───────────────
    ax_acc = fig.add_subplot(gs[1, 2])
    cat_acc = np.mean(y_pred[y_test==0] == 0) * 100
    dog_acc = np.mean(y_pred[y_test==1] == 1) * 100
    ax_acc.bar(["Cat", "Dog"], [cat_acc, dog_acc], color=["royalblue","tomato"],
               edgecolor="white", width=0.5)
    ax_acc.set_ylim(0, 110)
    ax_acc.axhline(100, color="gray", ls="--", lw=0.8)
    for i, v in enumerate([cat_acc, dog_acc]):
        ax_acc.text(i, v+1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax_acc.set_title("Per-Class Accuracy")
    ax_acc.set_ylabel("Accuracy (%)")

    # ── G–K. Sample Predictions (bottom row) ─
    indices  = np.random.RandomState(0).choice(len(y_test), 6, replace=False)
    for k, idx in enumerate(indices):
        row = 2
        col = k % 3
        if k == 3:
            row = 3   # we only have 3 rows in GridSpec, use a nested split
        ax_img = fig.add_subplot(gs[2, k % 3]) if k < 3 else None

    # rebuild bottom row properly
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2, :])
    sample_indices = np.random.RandomState(7).choice(len(y_test), 6, replace=False)
    label_names    = ["Cat", "Dog"]
    for k, idx in enumerate(sample_indices):
        r, c   = divmod(k, 3)
        ax_img = fig.add_subplot(gs2[r, c])
        img    = X_test[idx].reshape(*IMG_SIZE, 3)
        ax_img.imshow(np.clip(img, 0, 1))
        ax_img.axis("off")
        correct = y_pred[idx] == y_test[idx]
        color   = "green" if correct else "red"
        conf    = y_proba[idx] if y_pred[idx]==1 else 1-y_proba[idx]
        ax_img.set_title(
            f"Pred: {label_names[y_pred[idx]]}  ({conf*100:.0f}%)\nTrue: {label_names[y_test[idx]]}",
            color=color, fontsize=8
        )

    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✓ Saved → results.png")


# ══════════════════════════════════════════════
# BONUS — PREDICT A SINGLE NEW IMAGE
# ══════════════════════════════════════════════
def predict_single_image(model, image_path):
    """
    Call this after training to predict any image file.

    Example:
        model = joblib.load("svm_model.pkl")
        result = predict_single_image(model, "my_pet.jpg")
        print(result)
    """
    img = cv2.imread(image_path)
    if img is None:
        return f"Error: cannot read '{image_path}'"
    img   = cv2.resize(img, IMG_SIZE)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x     = img.flatten().astype(np.float32) / 255.0
    proba = model.predict_proba([x])[0]
    label = "Dog" if proba[1] > 0.5 else "Cat"
    conf  = max(proba) * 100
    return f"Prediction: {label}  ({conf:.1f}% confidence)"


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "═"*55)
    print("  Task-03  |  SVM Image Classifier: Cats vs Dogs")
    print("═"*55)

    # 1. Load
    X, y = load_images(DATASET_PATH, IMG_SIZE, MAX_IMAGES)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Evaluate
    y_pred, y_proba, acc = evaluate_model(model, X_test, y_test)

    # 5. Plot everything
    plot_all(model, X_test, y_test, y_pred, y_proba, acc)

    print(f"\n{'═'*55}")
    print(f"  ✅  DONE!  Final Accuracy: {acc*100:.2f}%")
    print(f"  Model saved to: {MODEL_FILE}")
    print(f"  Plots saved to: results.png")
    print(f"{'═'*55}")

    # Quick demo of single-image prediction
    print("""
  ─── To predict a new image later ───────────────────
  import joblib
  from svm_cats_dogs_COMPLETE import predict_single_image

  model = joblib.load("svm_model.pkl")
  print(predict_single_image(model, "your_image.jpg"))
  ─────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()