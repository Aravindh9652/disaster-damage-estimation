# Validation Report

**Generated:** 2026-11-04
**Target:** `C:\Users\vajja\Epics_Project\Mini_Project\disaster-damage-estimation`
**Overall:** **PARTIAL / BLOCKED** — code-level checks and demo UI routes were exercised, but PostgreSQL persistence and trained inference could not be verified in this environment.

## Environment and configuration inspection

| Area | Finding |
|---|---|
| Compose | `docker-compose.yml` defines `postgis/postgis:16-3.4`, persistent database volume, `pg_isready` health check, and API startup gated on the healthy database. Compose configuration parses successfully. |
| Migrations | `ml/persistence.py::initialize_database()` creates the PostGIS extension and `schema_migrations`, then applies sorted `ml/migrations/NNN_*.sql` files once. `001_assessments.sql` creates the assessment and review audit tables, JSONB evidence fields, optional SRID 4326 points, constraints, and indexes. |
| Persistence | PostgreSQL holds assessment/review metadata; original images and masks are files under `data/assessments/`. Compose bind-mounts that directory for the API. Review updates the current assessment and appends an audit row in one transaction. |
| API | Relevant routes: `POST /api/analyze`, `POST /api/analyze-pair`, `GET /api/assessments`, `GET /api/assessments/{id}`, `POST /api/assessments/{id}/review`, `GET /api/reviews`, `GET /api/assessments/map`, image routes, and comparison routes. API lifespan applies migrations if `DATABASE_URL` is configured. |
| Checkpoints | Expected paths are `models\resnet50_best.pt` (ResNet50) and `models\unet_damage.pt` (U-Net). Both `Test-Path` checks returned `False`; `models` contains only `.gitkeep`. No checkpoints were downloaded or synthesized. |
| Map geometry | The catalog contains illustrative polygon coordinates, transformed around approximate event-city anchors at a synthetic pixel scale. These are **not verified disaster locations or original xBD georeferencing**. The map page labels the catalog and its classes as synthetic; API GeoJSON includes an explicit note. Persisted analysis points are distinct and only created from caller-supplied coordinates. |

### Windows PowerShell: start PostGIS and apply migrations

Run from the repository root with Docker Desktop running. API startup invokes the migration runner automatically:

```powershell
Set-Location 'C:\Users\vajja\Epics_Project\Mini_Project\disaster-damage-estimation'
$env:POSTGRES_DB = 'damagemap'
$env:POSTGRES_USER = 'damagemap'
$env:POSTGRES_PASSWORD = 'local-development-only'
docker compose up -d db
if ($LASTEXITCODE -ne 0) { throw 'PostGIS did not start.' }

do {
    docker compose exec -T db pg_isready -U $env:POSTGRES_USER -d $env:POSTGRES_DB *> $null
    $databaseReady = $LASTEXITCODE -eq 0
    if (-not $databaseReady) { Start-Sleep -Seconds 2 }
} until ($databaseReady)

$env:DATABASE_URL = 'postgresql+psycopg://damagemap:local-development-only@127.0.0.1:5432/damagemap'
python -m uvicorn ml.api:app --host 127.0.0.1 --port 8000
```

In another PowerShell terminal, verify the migration and PostGIS extension:

```powershell
Set-Location 'C:\Users\vajja\Epics_Project\Mini_Project\disaster-damage-estimation'
docker compose exec -T db psql -U damagemap -d damagemap -c 'SELECT version, applied_at FROM schema_migrations ORDER BY version;'
docker compose exec -T db psql -U damagemap -d damagemap -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'postgis';"
```

To run both Compose services instead, `docker compose up --build -d` starts the API after the database health check; API startup applies migrations. The UI can be started separately with `Set-Location web; npm run dev`.

For a host-run API after starting the Compose database, set the same `DATABASE_URL` above and start Uvicorn. To explicitly run the migration runner without starting Uvicorn, use:

```powershell
$env:DATABASE_URL = 'postgresql+psycopg://damagemap:local-development-only@127.0.0.1:5432/damagemap'
python -c 'from ml.persistence import initialize_database; initialize_database()'
```

These are reproduction instructions, not claims that database startup or migrations passed in this run.

## Executed tests and results

| Test / command | Actual result |
|---|---|
| `docker compose config --quiet` | **PASS**, exit code 0. |
| `docker info` | **BLOCKED**, Docker daemon unavailable: named pipe `//./pipe/dockerDesktopLinuxEngine` was not found. |
| `docker compose up -d db` | **BLOCKED**, exit code 1; failed to connect to the Docker Desktop Linux engine. |
| `Test-NetConnection -ComputerName 127.0.0.1 -Port 5432 -InformationLevel Quiet` | **BLOCKED**, returned `False`; no host PostgreSQL listener. |
| Host migration attempt: `$env:DATABASE_URL = 'postgresql+psycopg://damagemap:local-development-only@127.0.0.1:5432/damagemap'; python -c "from ml.persistence import initialize_database; initialize_database()"` | **UNVERIFIED**; blocked waiting for an unavailable database connection and was stopped after 60 seconds. No migration was confirmed applied. |
| `python -m unittest ml.test_decision ml.test_api_contract` | **PASS**, 15 tests, exit code 0. Includes separate model-readiness disclosure, no-fake-prediction behavior when ResNet is missing, disagreement decision rules, synthetic map disclosure, and explicit unavailable history/review routes without a database. |
| From `ml`: `python test_remaining.py` | **PASS**, output `ok`, exit code 0. |
| From `ml`: `python test_metrics.py` | **PASS**, output `ok`, exit code 0. |
| From `web`: `npm run build` | **PASS**, exit code 0; Next.js compile, TypeScript, and static page generation completed. Non-fatal warning: parent `C:\Users\vajja\package-lock.json` is outside the repo. |
| From `web`: `npm run lint` | **PASS**, exit code 0. |
| `git diff --check` | **PASS**, exit code 0. |
| `npx playwright install chromium` | **PASS**, exit code 0; downloaded Chromium and headless-shell build 1248. The separate `appmod-debug-app-in-browser` runner could not launch because it expected a different, absent `chromium_headless_shell-1200` executable. The integrated browser was used for the page checks below. |
| Browser: `/map` | **PASS for rendering/disclosure only**; page showed “Damage map · synthetic catalog” and stated polygons, classes, and city anchors are synthetic, not verified coordinates. The saved-assessment layer correctly reported that PostGIS/API persistence was unavailable. |
| Browser: `/priority` | **PASS for rendering/disclosure only**; page showed “Synthetic demo ranking”, documented the project score, separate uncertainty, and stated the values are demo catalog data. Not a real assessment. |
| Browser: `/run` | **PASS for form rendering only**; page requires trained ResNet50 weights for inference. |
| Browser: `/review` | **PASS for unavailable-state rendering**; after the request settled, it displayed the PostGIS/database unavailable error rather than an empty queue. |
| Browser: `/history`, Apply filters | **PASS for unavailable-state rendering**; displayed the PostGIS/database unavailable error. No persisted history was retrieved. |

### Live API probes

The API and Next.js development server were started locally for these probes and then stopped.

| Request | Actual result |
|---|---|
| `GET /api/health` | **200**; `status=catalog-only`, `classifier=false`, `segmentation=change-or-texture`, `persistence=disabled`; returned the two expected absolute checkpoint paths. |
| `GET /api/damage-map` | **200**; 184 catalog features; note says the coordinates are not original xBD georeferencing and nothing is stored. |
| `GET /api/priority-zones` | **200**; four synthetic zones, `stored=false`, configured scoring formula returned. |
| `POST /api/analyze` with an in-memory 32×32 PNG probe | **503**, exact response: `ResNet50 checkpoint not found at resnet50_best.pt. Train or copy the weights, then restart.` No prediction was returned. |
| `GET /api/assessments`, `GET /api/reviews`, `GET /api/assessments/map` | Each **503**, exact detail indicates `DATABASE_URL` and a migrated PostGIS database are required. |

The API probe image was test input only; it was not assigned a prediction. Existing decision tests confirm that comparison is unavailable unless a trained U-Net produced the segmentation.

## Requested workflow coverage

| Workflow | Result |
|---|---|
| Create, retrieve, list assessments | **UNVERIFIED / BLOCKED**: persistence is disabled and no PostGIS server is reachable. Image analysis is additionally blocked by the missing ResNet50 checkpoint, so no assessment could be created legitimately. |
| Save reviewer correction and retrieve it after backend restart | **UNVERIFIED / BLOCKED**: requires a created persisted assessment and a working database. Neither condition was available. |
| ResNet50 readiness | **UNAVAILABLE**, independently checked: expected checkpoint absent, health reports `classifier=false`, actual analysis returns 503. |
| U-Net readiness | **UNAVAILABLE**, independently checked: expected checkpoint absent, health reports `segmentation=change-or-texture` and emits the U-Net warning. No U-Net prediction or comparison was fabricated. |
| Model disagreement | Decision rules are unit-tested; actual inference-time disagreement is **UNVERIFIED / BLOCKED** because neither model checkpoint is present. |
| Review queue | Route and UI error handling were exercised; populated queue and review submission are **UNVERIFIED / BLOCKED** by the database and model prerequisites. |
| Priority score | Scoring formula/bounds covered by regression tests; priority endpoint and synthetic priority page rendered. Real inference-derived score is **UNVERIFIED / BLOCKED** by missing ResNet50. |
| Map display | Synthetic catalog map rendered with disclosure. Persisted assessment map is **UNVERIFIED / BLOCKED** by PostGIS; saved points require explicitly supplied coordinates. |
| Assessment history | UI filter and explicit database error rendered; history persistence, filtering of saved records, and restart survival are **UNVERIFIED / BLOCKED**. |

## Reproduction steps after blockers are resolved

1. Start Docker Desktop and run the PowerShell commands above. Verify `schema_migrations` and the PostGIS extension with `psql`.
2. Provide the project-trained `models\resnet50_best.pt` checkpoint. Restart the API and confirm `/api/health` reports `classifier=true`. For actual disagreement testing, also provide the project-trained `models\unet_damage.pt` and confirm `segmentation=unet`.
3. Use a legitimate, locally available building crop (not generated demo imagery) with `POST /api/analyze` or `/api/analyze-pair`. Supply coordinates only when verified and intended for map display.
4. Record the returned assessment ID; verify `GET /api/assessments/{id}`, filtered `GET /api/assessments`, image routes, `/api/reviews`, and `/api/assessments/map`.
5. Submit a review correction with `POST /api/assessments/{id}/review`, verify the human result and review audit record, restart the API (`docker compose restart api` for Compose), then repeat the GET requests to verify database persistence across restart.
6. Exercise the `/run`, `/review`, `/history`, `/priority`, and `/map` browser flows against those saved real assessments. Keep the synthetic catalog layer labeled as demo and separate from saved assessment points.

## Unresolved issues / blockers

1. Docker Desktop is not running or its Linux-engine named pipe is unavailable. PostgreSQL/PostGIS migration and restart persistence testing could not be performed.
2. Both expected model checkpoints are absent. Real image classification, U-Net segmentation, and actual model disagreement cannot be tested; the API correctly rejects classification instead of fabricating a result.
3. No trained, real-image end-to-end flow could therefore reach persistence, reviewer correction, or history. Those behaviors remain unverified, not passed.
4. The standalone browser-debug tool expected a different Playwright Chromium revision than the one installed. Integrated browser page inspection worked for the reported pages, but no automated browser spec was authored or replayed.
