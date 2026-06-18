# Vector Store Maintenance on Kubernetes

## Summary

The deployed URBIND backend does **not** auto-refresh the vector store in Kubernetes.
It only detects whether `/data/chroma` is stale and blocks vector-backed runs until an
operator runs the maintenance workflow.

## Why automatic updater Jobs were disabled

The current cluster storage layout makes automatic updater Jobs a bad fit:

- The backend PVC `urbind-query-mechanism-backend-output` uses `ReadWriteOnce`.
- The backend pod mounts that PVC at `/data`.
- A second pod on another node cannot attach the same PVC at the same time.
- When a one-off updater Job was created automatically, Kubernetes left the Job pod in
  `Pending` / `ContainerCreating` with `FailedAttachVolume` because the backend pod was
  already using the PVC.

Even if the Job could mount the volume, Chroma would still be reading and writing the
same folder in place, which is a poor fit for unattended live updates.

## Historical note about RBAC

We briefly added RBAC and runtime Job creation so the backend could create
vector-store updater Jobs automatically.

That RBAC was removed from the repo because:

- the deployed maintenance workflow no longer relies on automatic updater Jobs
- the real blocker was the PVC/storage topology, not missing Kubernetes permissions
- keeping the RBAC manifest around would suggest an automatic-job path that we do not
  want to re-enable casually

If we revisit automatic Kubernetes updater Jobs in the future, first revisit storage
architecture, not RBAC.

## Current supported workflow

Use the maintenance script from the repo root:

```bash
bash scripts/update_vector_store_maintenance.sh
```

What it does:

1. scales `deployment/urbind-query-mechanism-backend` to `0`
2. deletes any previous `urbind-query-mechanism-build-vector-index` Job
3. applies `k8s/backend-build-vector-index-job.yml`
4. prints live Job/Pod status while waiting, and fails early when the Job pod is stuck in `Pending` / `ContainerCreating`
5. surfaces `FailedAttachVolume` clearly when the PVC is still mounted elsewhere
6. scales the backend back to `1`
7. waits for rollout readiness

## UI / API behavior

With `VECTOR_STORE_AUTO_UPDATE_ON_RUN=false`:

- startup still performs a dry-run freshness check
- each vector-backed run submission performs a dry-run freshness check
- if the index is stale, the backend returns a stale status instead of launching a Job
- the frontend shows a manual-maintenance banner with the maintenance command

## If we ever revisit automatic updates

Do not start by reintroducing the old RBAC manifest alone.

First confirm a storage design that supports the desired behavior, for example:

- a maintenance-only update flow with backend downtime, or
- a different shared-storage approach plus a publish/swap model for the vector store

Without that storage decision, automatic updater Jobs are likely to fail again.
