# Streamlit Cloud Deployment Guide

## Why You're Seeing "Not Connected to GitHub" Error

Your repository **is** connected to GitHub and is **public**, so the issue is likely with Streamlit Cloud's GitHub app permissions.

## Step-by-Step Deployment Instructions

### 1. **Authorize Streamlit to Access Your Repository**

1. Go to https://share.streamlit.io/
2. Click **Sign in with GitHub**
3. If already signed in, click your profile → **Settings** → **Connected accounts**
4. Click **Reconnect** or **Authorize** for GitHub
5. Make sure to grant access to the `Energy_Market_Forecasting_project` repository

### 2. **Deploy Your App**

Option A: **Using the Web Interface**
1. Go to https://share.streamlit.io/
2. Click **New app**
3. Select **Deploy a public app from GitHub**
4. Fill in:
   - **Repository:** `TyredMayfly/Energy_Market_Forecasting_project`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **Deploy!**

Option B: **Using Direct URL**
Visit this URL directly:
```
https://share.streamlit.io/deploy?repository=TyredMayfly/Energy_Market_Forecasting_project&branch=main&mainModule=streamlit_app.py
```

### 3. **Configure API Secrets** (CRITICAL!)

After deployment starts, you **must** add your API keys:

1. In Streamlit Cloud, go to your app
2. Click **⋮** (three dots menu) → **Settings**
3. Click **Secrets** tab
4. Paste your API keys in TOML format:

```toml
# ENTSO-E Transparency Platform API Key
ENTSOE_API_KEY = "your_actual_entsoe_api_key_here"

# Meteosource Weather API Key
METEOSOURCE_API_KEY = "your_actual_meteosource_api_key_here"

# TenneT Settlement Prices API Key
TENNET_API_KEY = "your_actual_tennet_api_key_here"
```

5. Click **Save**
6. Your app will automatically restart with the secrets

### 4. **Verify Deployment**

Your app should be accessible at:
```
https://[your-app-name].streamlit.app
```

## Troubleshooting

### "Repository not found" or "Not connected"
- **Solution:** Revoke and re-authorize Streamlit's GitHub app:
  1. Go to https://github.com/settings/installations
  2. Find **Streamlit**
  3. Click **Configure**
  4. Under "Repository access", select "All repositories" or specifically select `Energy_Market_Forecasting_project`
  5. Save and try deploying again

### App crashes on startup
- **Cause:** Missing API keys
- **Solution:** Add secrets as shown in Step 3

### "Module not found" errors
- **Cause:** Missing dependencies
- **Solution:** Verify all packages are in `requirements.txt` (already configured correctly)

### Data files not found
- **Status:** Your CSV data files are committed to the repo, so they'll be available
- **Note:** Streamlit Cloud has a 1GB storage limit

## Alternative Deployment Options

If Streamlit Cloud continues to have issues:

1. **Render** (Free tier available): https://render.com/
2. **Railway** (Free tier with usage limits): https://railway.app/
3. **Google Cloud Run** (Pay-as-you-go, generous free tier)
4. **Heroku** (No longer has free tier)

## Current Repository Status

✅ Repository: `TyredMayfly/Energy_Market_Forecasting_project`  
✅ Visibility: **Public**  
✅ Branch: `main`  
✅ Streamlit app file: `streamlit_app.py` (in root)  
✅ Dependencies: `requirements.txt` (configured)  
✅ Data files: Included in repo  
✅ Secrets: Template created in `.streamlit/secrets.toml` (gitignored)

## Need Help?

If you still can't deploy:
1. Check GitHub app permissions at: https://github.com/settings/installations
2. Try the direct URL deployment method (Option B above)
3. Check Streamlit Community forum: https://discuss.streamlit.io/
