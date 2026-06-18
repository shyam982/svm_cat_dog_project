# Push Guide — How to Push to GitHub

This folder contains:
- **cat_dogs.py** — Main SVM classifier script
- **README.md** — Project documentation
- **requirements.txt** — Python dependencies
- **results.png** — Sample output visualization
- **svm_model.pkl** — Trained model file
- **.gitignore** — Files to exclude from git

## Step-by-Step Push Instructions

### 1. Install Git (if not already installed)
Download and install from: https://git-scm.com/download/win

### 2. Open Command Prompt (cmd)
Navigate to the project folder:
```bash
cd "c:\Users\vhinb\OneDrive\Desktop\cat_dog_svm_project"
```

### 3. Configure Git (first time only)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 4. Initialize Git Repository
```bash
git init
```

### 5. Add All Files
```bash
git add .
```

### 6. Create Initial Commit
```bash
git commit -m "Initial commit: SVM cats vs dogs classifier with README and requirements"
```

### 7. Set Main Branch
```bash
git branch -M main
```

### 8. Add Remote Repository
```bash
git remote add origin https://github.com/shyam982/svm_cat_dog_project.git
```
*(If remote already exists, use: `git remote set-url origin https://github.com/shyam982/svm_cat_dog_project.git`)*

### 9. Push to GitHub
```bash
git push -u origin main
```

When prompted, authenticate using:
- **Personal Access Token** (recommended) — create at https://github.com/settings/tokens
- **GitHub credentials** (username & personal access token as password)

## Troubleshooting

**Remote already exists?**
```bash
git remote -v
git remote remove origin
git remote add origin https://github.com/shyam982/svm_cat_dog_project.git
```

**Want to update an existing commit?**
```bash
git add .
git commit --amend --no-edit
git push -f origin main
```

**Check git status:**
```bash
git status
```

## Files Summary

| File | Purpose |
|------|---------|
| `cat_dogs.py` | Main classifier with SVM pipeline |
| `README.md` | Project overview and usage |
| `requirements.txt` | Python packages needed |
| `results.png` | Sample output plot |
| `svm_model.pkl` | Trained scikit-learn model |
| `.gitignore` | Files excluded from version control |

---

**Note:** If you'd prefer not to push the large `svm_model.pkl` and `results.png` files, uncomment the last two lines in `.gitignore` before pushing.
