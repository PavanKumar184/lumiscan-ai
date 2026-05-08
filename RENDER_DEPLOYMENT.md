# Deploying LumiScan AI On Render

This project is prepared for Render deployment as a Python Web Service.

## Files Added For Render

- `Procfile`: tells Render how to start the Flask app using Gunicorn.
- `runtime.txt`: requests Python 3.11.9.
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

6. Choose an instance type.

   Recommended:

   ```text
   At least 1 GB RAM
   ```

   The free tier may be slow or may fail because PyTorch and the DenseNet model need memory.

7. Click:

   ```text
   Create Web Service
   ```

8. Wait for Render to build and deploy.

9. After deployment, Render will provide a URL similar to:

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

   Upgrade the Render instance plan.

2. **PyTorch installation error**

   Confirm `requirements.txt` contains the CPU PyTorch extra index:

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
