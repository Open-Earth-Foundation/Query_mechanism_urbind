# Manual Deployment Guide (GHCR + EKS, No GitHub Actions)

This guide deploys backend + frontend to OEF dev EKS using direct commands only.
No PowerShell environment variables are required.

## 1. Prerequisites

- Docker with `buildx`
- AWS CLI authenticated
- `kubectl`
- GHCR PAT (classic) with package permissions

Do not store secrets in repository files.

## 2. Login to GHCR

```powershell
echo "<your_github_pat>" | docker login ghcr.io -u "<your_github_username>" --password-stdin
```

## 3. Build and Push Backend Image

Windows/Linux (`amd64`):

```powershell
docker buildx build --platform linux/amd64 -f backend/Dockerfile -t ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:dev --push .
```

## 4. Build and Push Frontend Image

Windows/Linux (`amd64`):

```powershell
docker buildx build --platform linux/amd64 -f frontend/Dockerfile --build-arg NEXT_PUBLIC_API_BASE_URL=https://urbind-query-mechanism-api.openearth.dev -t ghcr.io/open-earth-foundation/query_mechanism_urbind-frontend:dev --push ./frontend
```

## 5. Make Packages Public in GHCR

1. Open `https://github.com/orgs/Open-Earth-Foundation/packages`
2. Open backend and frontend packages
3. Set visibility to `Public`

## 6. Connect to EKS

```powershell
kubectl get nodes
```

## 7. Create/Update Backend Secret

```powershell
kubectl create secret generic urbind-query-mechanism-backend-secrets --from-literal=OPENROUTER_API_KEY="<your_openrouter_api_key>" --dry-run=client -o yaml | kubectl apply -f -
```

## 8. Deploy Kubernetes Resources

```powershell
kubectl apply -f k8s/backend-configmap.yml
kubectl apply -f k8s/backend-pvc.yml
kubectl apply -f k8s/backend-deployment.yml
kubectl apply -f k8s/backend-service.yml
kubectl apply -f k8s/frontend-deployment.yml
kubectl apply -f k8s/frontend-service.yml
```

Note: the backend deployment uses `strategy: Recreate` because it mounts a `ReadWriteOnce`
PVC (`urbind-query-mechanism-backend-output`). This avoids `Multi-Attach` rollout failures.

## 9. Pin Deployment Images

```powershell
kubectl set image deployment/urbind-query-mechanism-backend backend=ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:dev
kubectl set image deployment/urbind-query-mechanism-frontend frontend=ghcr.io/open-earth-foundation/query_mechanism_urbind-frontend:dev
```

## 10. Wait for Rollout

```powershell
kubectl rollout status deployment/urbind-query-mechanism-backend
kubectl rollout status deployment/urbind-query-mechanism-frontend
```

## 11. Verify Pods and Services

```powershell
kubectl get pods
kubectl get svc
kubectl describe service/urbind-query-mechanism-backend-service
kubectl describe service/urbind-query-mechanism-frontend-service
```

Check that `Endpoints` are populated.

## 12. Add Ingress Rules

```powershell
kubectl edit ingress
```

Add:

```yaml
- host: urbind-query-mechanism.openearth.dev
  http:
    paths:
      - backend:
          service:
            name: urbind-query-mechanism-frontend-service
            port:
              number: 3000
        path: /
        pathType: Prefix
- host: urbind-query-mechanism-api.openearth.dev
  http:
    paths:
      - backend:
          service:
            name: urbind-query-mechanism-backend-service
            port:
              number: 8000
        path: /
        pathType: Prefix
```

## 13. Test Deployment

```powershell
curl https://urbind-query-mechanism-api.openearth.dev/
curl https://urbind-query-mechanism-api.openearth.dev/healthz
```

Open:

- `https://urbind-query-mechanism.openearth.dev`

Expected:

- Frontend loads
- Backend root `/` responds with service health JSON

## 14. Update Release (Manual Repeat)

```powershell
docker buildx build --platform linux/amd64 -f backend/Dockerfile -t ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:dev --push .
docker buildx build --platform linux/amd64 -f frontend/Dockerfile --build-arg NEXT_PUBLIC_API_BASE_URL=https://urbind-query-mechanism-api.openearth.dev -t ghcr.io/open-earth-foundation/query_mechanism_urbind-frontend:dev --push ./frontend

kubectl set image deployment/urbind-query-mechanism-backend backend=ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:dev
kubectl set image deployment/urbind-query-mechanism-frontend frontend=ghcr.io/open-earth-foundation/query_mechanism_urbind-frontend:dev

kubectl rollout status deployment/urbind-query-mechanism-backend
kubectl rollout status deployment/urbind-query-mechanism-frontend
```

## 15. GitHub Actions (Automated Deploy)

This repository includes `.github/workflows/develop.yml`.

It does:

1. Build and push backend image to GHCR:
   `ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:<sha>` and `:dev`
2. Build and push frontend image to GHCR:
   `ghcr.io/open-earth-foundation/query_mechanism_urbind-frontend:<sha>` and `:dev`
3. Connect to EKS and apply manifests from `k8s/`
4. Update deployment images and wait for rollout

Required GitHub repository secrets:

- `AWS_ACCESS_KEY_ID_EKS_DEV_USER`
- `AWS_SECRET_ACCESS_KEY_EKS_DEV_USER`
- `EKS_DEV_NAME`
- `OPENROUTER_API_KEY`

Optional GitHub repository variables:

- `EKS_DEV_REGION` (default: `us-east-1`)
- `FRONTEND_API_BASE_URL` (default: `https://urbind-query-mechanism-api.openearth.dev`)

How to use:

1. Open a PR to `main` (tests run), then merge to `main` to trigger build + deploy.
2. Or run manually in GitHub UI: `Actions` -> `Dev - Test, Build, and Deploy urbind-query-mechanism` -> `Run workflow`.
