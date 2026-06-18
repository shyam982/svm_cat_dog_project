# Git Push Guide

## How to Push Your Code to GitHub

### Prerequisites
- Git installed on your machine
- GitHub account
- Repository created on GitHub

### Steps to Push

1. **Initialize Git (if not already done)**
   ```bash
   git init
   ```

2. **Add all files**
   ```bash
   git add .
   ```

3. **Create initial commit**
   ```bash
   git commit -m "Initial commit"
   ```

4. **Add remote repository**
   ```bash
   git remote add origin <your-github-repo-url>
   ```

5. **Push to GitHub**
   ```bash
   git push -u origin main
   ```

### Subsequent Pushes
Once set up, you can simply use:
```bash
git push
```

### Useful Git Commands
- `git status` - Check current status
- `git log` - View commit history
- `git branch` - View branches
- `git pull` - Pull latest changes from remote

### Troubleshooting
- If push fails, ensure your remote URL is correct: `git remote -v`
- Make sure you have the latest changes: `git pull` before pushing
- For authentication issues, use SSH keys or personal access tokens

---
For more information, visit [GitHub Docs](https://docs.github.com/en/get-started/using-git)
