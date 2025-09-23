# Deployment Guide

Panduan lengkap untuk deploy API ke Render dan Azure.

## 🚀 Deployment ke Render

### Persiapan
1. Push kode ke GitHub repository
2. Pastikan file `render.yaml` sudah ada dan dikonfigurasi dengan benar
3. Login ke [Render Dashboard](https://dashboard.render.com/)

### Langkah Deploy
1. **Buat Web Service Baru**
   - Klik "New" → "Web Service"
   - Connect GitHub repository
   - Pilih repository `ta-api-sayadi`

2. **Konfigurasi Service**
   - **Name**: `ta-api-sayadi`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables**
   ```
   MODEL_DIR=model
   PYTHONPATH=.
   API_KEY=your-secret-api-key
   ```

4. **Deploy**
   - Klik "Create Web Service"
   - Tunggu proses build dan deploy selesai
   - URL akan tersedia di dashboard

### Auto Deploy
Render akan otomatis redeploy setiap kali ada push ke branch main/master.

---

## ☁️ Deployment ke Azure

### Opsi 1: Azure App Service (Recommended)

#### Persiapan
1. Install Azure CLI: `az login`
2. Buat Resource Group dan App Service Plan

#### Deploy Manual
```bash
# Login ke Azure
az login

# Buat resource group
az group create --name rg-ta-api-sayadi --location "Southeast Asia"

# Buat App Service Plan
az appservice plan create --name plan-ta-api-sayadi --resource-group rg-ta-api-sayadi --sku B1 --is-linux

# Buat Web App
az webapp create --resource-group rg-ta-api-sayadi --plan plan-ta-api-sayadi --name ta-api-sayadi --runtime "PYTHON:3.13" --deployment-local-git

# Set startup command
az webapp config set --resource-group rg-ta-api-sayadi --name ta-api-sayadi --startup-file "uvicorn main:app --host 0.0.0.0 --port 8000"

# Set environment variables
az webapp config appsettings set --resource-group rg-ta-api-sayadi --name ta-api-sayadi --settings MODEL_DIR=model PYTHONPATH=. API_KEY=your-secret-api-key

# Deploy kode
az webapp deployment source config-local-git --name ta-api-sayadi --resource-group rg-ta-api-sayadi
git remote add azure <git-url-from-previous-command>
git push azure main
```

#### Deploy dengan Azure DevOps
1. Push kode ke Azure DevOps repository
2. Buat pipeline menggunakan `azure-pipelines.yml`
3. Konfigurasi service connection ke Azure subscription
4. Run pipeline

### Opsi 2: Azure Container Instances

```bash
# Build dan push Docker image
docker build -t ta-api-sayadi .
docker tag ta-api-sayadi your-registry.azurecr.io/ta-api-sayadi:latest
docker push your-registry.azurecr.io/ta-api-sayadi:latest

# Deploy ke ACI
az container create --resource-group rg-ta-api-sayadi --name ta-api-sayadi --image your-registry.azurecr.io/ta-api-sayadi:latest --cpu 1 --memory 1 --ports 8000 --environment-variables MODEL_DIR=model PYTHONPATH=. API_KEY=your-secret-api-key
```

### Opsi 3: Azure Kubernetes Service (AKS)

1. Buat AKS cluster
2. Deploy menggunakan Kubernetes manifests
3. Konfigurasi ingress dan load balancer

---

## 🔧 Konfigurasi Environment Variables

### Required Variables
- `API_KEY`: Secret key untuk autentikasi
- `MODEL_DIR`: Path ke folder model (default: `model`)
- `PYTHONPATH`: Python path (default: `.`)

### Optional Variables
- `ENVIRONMENT`: `production` atau `development`
- `LOG_LEVEL`: `INFO`, `DEBUG`, `WARNING`, `ERROR`
- `MAX_WORKERS`: Jumlah worker processes

---

## 🏥 Health Check

Semua deployment menggunakan endpoint `/health` untuk health check:
- **URL**: `https://your-app-url/health`
- **Method**: GET
- **Response**: `{"status": "healthy"}`

---

## 📊 Monitoring

### Render
- Built-in metrics di dashboard
- Logs tersedia di dashboard
- Alerts via email

### Azure
- Application Insights untuk monitoring
- Azure Monitor untuk metrics
- Log Analytics untuk logs

---

## 🔒 Security

1. **API Key**: Selalu gunakan environment variable
2. **HTTPS**: Aktifkan SSL/TLS
3. **CORS**: Konfigurasi sesuai kebutuhan
4. **Rate Limiting**: Implementasi jika diperlukan

---

## 🐛 Troubleshooting

### Common Issues

1. **Model tidak ditemukan**
   - Pastikan folder `model/` ada di repository
   - Check environment variable `MODEL_DIR`

2. **Import error**
   - Pastikan semua dependencies ada di `requirements.txt`
   - Check Python version compatibility

3. **Port binding error**
   - Pastikan menggunakan `$PORT` environment variable
   - Default port 8000 untuk local development

4. **Memory issues**
   - Upgrade plan jika diperlukan
   - Optimize model loading

### Logs

**Render**: Dashboard → Service → Logs
**Azure**: Portal → App Service → Log stream

---

## 📝 Notes

- Render free tier memiliki limitasi (sleep setelah 15 menit idle)
- Azure free tier memiliki quota bulanan
- Pastikan model files tidak terlalu besar untuk deployment
- Gunakan `.dockerignore` untuk mengoptimalkan Docker build

---

## 🔗 Useful Links

- [Render Documentation](https://render.com/docs)
- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)