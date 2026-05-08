# Deploying LumiScan AI On Render

This project is prepared for Render deployment as a Python Web Service.

## Files Added For Render

- `Procfile`: tells Render how to start the Flask app using Gunicorn.
- `.python-version`: requests Python 3.11.9.
- `runtime.txt`: kept as a fallback Python version hint.
- `requirements.txt`: contains the production dependencies, including CPU PyTorch.
- `app.py`: reads Render's dynamic `PORT` environment variable.

## Render Deployment Steps

1. Go to Render:

   ```text
   https://render.com
   ```

2. Sign in with GitHub or connect your GitHub account.

3. Click:

   ```text
   New + -> Web Service
   ```

4. Select the GitHub repository:

   ```text
   PavanKumar184/lumiscan-ai
   ```

5. Configure the service:

   ```text
   Name: lumiscan-ai
   Runtime: Python 3
   Region: Choose nearest region
   Branch: main
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app --workers 1 --threads 2 --timeout 180
   ```

6. Add this environment variable in Render:

   ```text
   PYTHON_VERSION=3.11.9
   TORCH_NUM_THREADS=1
   ```

   This is important because Render's newer default Python version may be `3.14.x`, and the pinned PyTorch CPU wheel used by this project is not available for Python 3.14.

   Grad-CAM heatmap generation is disabled by default on Render to avoid memory crashes during prediction. If you upgrade to a larger instance and want heatmaps online, add:

   ```text
   ENABLE_GRADCAM=true
   ```

7. Choose an instance type.

   Recommended:

   ```text
   At least 1 GB RAM
   ```

   The free tier may be slow or may fail because PyTorch and the DenseNet model need memory.

8. Click:

   ```text
   Create Web Service
   ```

9. Wait for Render to build and deploy.

10. After deployment, Render will provide a URL similar to:

   ```text
   https://lumiscan-ai.onrender.com
   ```

## Important Notes

- The app loads `best_dr_ultra_densenet121_nih.pth` at startup, so the first deployment/startup can take time.
- Render storage is ephemeral. Uploaded X-rays and Grad-CAM images may not persist permanently.
- The project is for educational/research demonstration and should not be used as a final clinical diagnosis tool.

## If Deployment Fails

Check these common issues:

1. **Out of memory**

   Keep Grad-CAM disabled or upgrade the Render instance plan. Grad-CAM uses more memory than normal model prediction because it runs a backward pass through the CNN.

2. **PyTorch installation error**

   Confirm Render is using Python 3.11.9. In the Render dashboard, set:

   ```text
   PYTHON_VERSION=3.11.9
   ```

   Also confirm `requirements.txt` contains the CPU PyTorch extra index:

   ```text
   --extra-index-url https://download.pytorch.org/whl/cpu
   ```

3. **App does not start**

   Confirm the start command is:

   ```bash
   gunicorn app:app --workers 1 --threads 2 --timeout 180
   ```

4. **Model missing**

   Confirm this file exists in the GitHub repository:

   ```text
   best_dr_ultra_densenet121_nih.pth
   ```
