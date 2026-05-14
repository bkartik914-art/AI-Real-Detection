# Azure Container Apps deployment (CPU, two containers)

This guide deploys the FastAPI backend and Next.js frontend as two Azure Container Apps using Azure Container Registry (ACR).

## Prereqs

- Azure CLI (`az`) installed and logged in
- Docker Desktop installed

## 1) Set variables (PowerShell)

```powershell
$RG = "ai-real-detect-rg"
$REGION = "southeastasia"  # Use: eastasia or southeastasia
$ACR = "airealdetectacr123" # Must be globally unique, lowercase
$ENV = "ai-real-env"
```

## 2) Create resource group + ACR

```powershell
az login
az group create -n $RG -l $REGION
az acr create -n $ACR -g $RG --sku Basic --admin-enabled true

$ACR_LOGIN = (az acr show -n $ACR --query "loginServer" -o tsv)
$ACR_USER = (az acr credential show -n $ACR --query "username" -o tsv)
$ACR_PASS = (az acr credential show -n $ACR --query "passwords[0].value" -o tsv)
```

## 3) Build + push backend image

From the repo root:

```powershell
docker build -f backend/Dockerfile -t $ACR_LOGIN/ai-real-backend:1.0.0 .
docker push $ACR_LOGIN/ai-real-backend:1.0.0
```

## 4) Create Container Apps environment

```powershell
az containerapp env create -g $RG -n $ENV -l $REGION
```

## 5) Deploy backend app

```powershell
az containerapp create -g $RG -n ai-real-backend --environment $ENV `
  --image $ACR_LOGIN/ai-real-backend:1.0.0 --ingress external --target-port 8000 `
  --registry-server $ACR_LOGIN --registry-username $ACR_USER --registry-password $ACR_PASS `
  --env-vars MODEL_DEVICE=cpu LOAD_MODEL_ON_STARTUP=false

$BACKEND_FQDN = (az containerapp show -g $RG -n ai-real-backend --query "properties.configuration.ingress.fqdn" -o tsv)
$BACKEND_URL = "https://$BACKEND_FQDN"
```

## 6) Build + push frontend image (backend URL baked in)

`NEXT_PUBLIC_API_URL` is compiled into the client bundle. If the backend URL changes, rebuild and redeploy the frontend image.

```powershell
docker build -f frontend/Dockerfile -t $ACR_LOGIN/ai-real-frontend:1.0.0 `
  --build-arg NEXT_PUBLIC_API_URL=$BACKEND_URL .
docker push $ACR_LOGIN/ai-real-frontend:1.0.0
```

## 7) Deploy frontend app

```powershell
az containerapp create -g $RG -n ai-real-frontend --environment $ENV `
  --image $ACR_LOGIN/ai-real-frontend:1.0.0 --ingress external --target-port 3000 `
  --registry-server $ACR_LOGIN --registry-username $ACR_USER --registry-password $ACR_PASS

$FRONTEND_FQDN = (az containerapp show -g $RG -n ai-real-frontend --query "properties.configuration.ingress.fqdn" -o tsv)
$FRONTEND_URL = "https://$FRONTEND_FQDN"
```

## 8) Update backend CORS to allow frontend

```powershell
az containerapp update -g $RG -n ai-real-backend --set-env-vars CORS_ORIGINS=$FRONTEND_URL
```

## Notes

- Backend listens on port 8000. Frontend listens on port 3000.
- `MODEL_DEVICE=cpu` forces CPU inference.
- You can scale with `--min-replicas` and `--max-replicas` on the Container Apps.
