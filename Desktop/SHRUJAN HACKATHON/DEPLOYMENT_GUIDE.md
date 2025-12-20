# 🚀 RAG Document QA System - Deployment Guide

## Overview
Your RAG Document QA System is a Streamlit application that can be deployed to various cloud platforms. This guide covers multiple deployment options, from easiest to most advanced.

## 📋 Prerequisites
- All dependencies installed (requirements.txt)
- Application tested locally
- OpenAI API key (for full functionality)
- Git repository (recommended)

## 🌐 Deployment Options

### 1. **Streamlit Cloud (Easiest) - RECOMMENDED** ⭐

**Best for:** Quick deployment, automatic scaling, free tier available

#### Steps:
1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   # Create repo on GitHub and push
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Environment Variables**
   - In Streamlit Cloud dashboard, go to your app settings
   - Add environment variable: `OPENAI_API_KEY=your_api_key_here`

#### Pros:
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ No server management
- ✅ Auto-scaling

#### Cons:
- ❌ Limited customization
- ❌ May have cold start delays

---

### 2. **Heroku (Popular Choice)**

**Best for:** Professional deployment, good performance

#### Steps:
1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Required Files**

   **requirements.txt** (already exists)

   **Procfile** (create new):
   ```
   web: streamlit run --server.port $PORT --server.headless true app.py
   ```

   **runtime.txt** (create new):
   ```
   python-3.12.0
   ```

   **.streamlit/config.toml** (create new):
   ```toml
   [server]
   headless = true
   port = 8501

   [browser]
   gatherUsageStats = false
   ```

3. **Deploy**
   ```bash
   heroku create your-app-name
   heroku config:set OPENAI_API_KEY=your_api_key_here
   git push heroku main
   ```

#### Pros:
- ✅ Good performance
- ✅ Custom domains
- ✅ Environment variables support
- ✅ Free tier available

#### Cons:
- ❌ Requires Heroku CLI setup
- ❌ May sleep on free tier

---

### 3. **Railway (Modern Alternative)**

**Best for:** Easy deployment, good performance, modern stack

#### Steps:
1. **Connect Repository**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway auto-detects Streamlit app

2. **Environment Variables**
   - Add `OPENAI_API_KEY` in Railway dashboard

3. **Deploy**
   - Railway automatically builds and deploys
   - Provides custom domain

#### Pros:
- ✅ Very easy setup
- ✅ Good performance
- ✅ Modern platform
- ✅ Generous free tier

---

### 4. **Render (Simple & Reliable)**

**Best for:** Reliable hosting, good for production

#### Steps:
1. **Create render.yaml**
   ```yaml
   services:
     - type: web
       name: rag-qa-system
       runtime: python3
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run --server.port $PORT --server.headless true app.py
       envVars:
         - key: OPENAI_API_KEY
           value: your_api_key_here
   ```

2. **Deploy**
   - Connect GitHub repository to Render
   - Render auto-deploys on git push

#### Pros:
- ✅ Reliable hosting
- ✅ Free tier available
- ✅ Custom domains
- ✅ Good documentation

---

### 5. **AWS EC2 (Advanced)**

**Best for:** Full control, enterprise deployments

#### Steps:
1. **Launch EC2 Instance**
   - Choose Ubuntu Server
   - t3.medium or larger recommended
   - Configure security groups (ports 22, 80, 443, 8501)

2. **Setup Server**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python and pip
   sudo apt install python3 python3-pip -y

   # Install dependencies
   pip3 install -r requirements.txt

   # Create systemd service
   sudo nano /etc/systemd/system/streamlit.service
   ```

3. **Systemd Service File**
   ```ini
   [Unit]
   Description=Streamlit App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/your-app
   ExecStart=/usr/bin/python3 -m streamlit run --server.port 8501 --server.headless true app.py
   Environment=OPENAI_API_KEY=your_api_key_here
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

4. **Start Service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start streamlit
   sudo systemctl enable streamlit
   ```

#### Pros:
- ✅ Full control
- ✅ Scalable
- ✅ Enterprise-ready

#### Cons:
- ❌ Complex setup
- ❌ Requires AWS knowledge
- ❌ Paid service

---

### 6. **Google Cloud Run (Serverless)**

**Best for:** Serverless deployment, auto-scaling

#### Steps:
1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.12-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .
   EXPOSE 8501

   CMD ["streamlit", "run", "--server.port", "8501", "--server.headless", "true", "app.py"]
   ```

2. **Deploy**
   ```bash
   gcloud run deploy rag-qa-system \
     --source . \
     --platform managed \
     --region us-central1 \
     --set-env-vars OPENAI_API_KEY=your_api_key_here
   ```

#### Pros:
- ✅ Auto-scaling
- ✅ Pay-per-use
- ✅ Serverless

---

## 🔧 Configuration Tips

### Environment Variables
```bash
# Required for full functionality
OPENAI_API_KEY=your_openai_api_key

# Optional optimizations
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Performance Optimization
- Use at least 2GB RAM for better performance
- Consider GPU instances for faster embeddings (if using cloud GPU)
- Enable caching for better user experience

### Security Considerations
- Never commit API keys to git
- Use environment variables for sensitive data
- Enable HTTPS in production
- Consider authentication for production use

## 🚀 Quick Start (Streamlit Cloud)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your repo
   - Set main file: `app.py`
   - Add `OPENAI_API_KEY` environment variable

3. **Access Your App**
   - Streamlit Cloud provides a public URL
   - Share with users or embed in websites

## 📊 Cost Comparison

| Platform | Free Tier | Paid Plan | Best For |
|----------|-----------|-----------|----------|
| Streamlit Cloud | 100 hours/month | $8/month | Quick prototyping |
| Heroku | 550 hours/month | $7/month | Professional apps |
| Railway | $5/month credit | $5/month | Modern deployments |
| Render | 750 hours/month | $7/month | Reliable hosting |
| AWS EC2 | - | $10-50/month | Enterprise |
| Google Cloud Run | 2M requests/month | Pay-per-use | Serverless |

## 🆘 Troubleshooting

### Common Issues:
1. **Memory Errors**: Upgrade to larger instance (2GB+ RAM)
2. **Timeout Errors**: Increase timeout limits in platform settings
3. **API Key Issues**: Ensure environment variables are set correctly
4. **Port Issues**: Use `$PORT` environment variable for dynamic ports

### Performance Tips:
- Pre-load models if possible
- Use caching decorators
- Optimize chunk sizes
- Consider CDN for static assets

## 📞 Support

- **Streamlit Cloud**: Check status.streamlit.io
- **Heroku**: heroku support
- **Railway**: Discord community
- **Render**: GitHub issues

---

**🎯 Recommendation**: Start with **Streamlit Cloud** for easiest deployment, then migrate to **Railway** or **Render** for production use.
