"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { levelClass, type PriorityLevel } from "@/lib/scoring";
import { AlertTriangle, FileImage, LoaderCircle, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";

const API_URL = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";
const LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"] as const;
const COLORS: Record<(typeof LABELS)[number], string> = {
  "no-damage": "#22c55e",
  "minor-damage": "#eab308",
  "major-damage": "#f97316",
  destroyed: "#ef4444",
};

type Result = {
  assessment_id?: string;
  review_status?: string;
  filename: string;
  prediction: (typeof LABELS)[number];
  confidence: number;
  probabilities: Record<(typeof LABELS)[number], number>;
  severity_score: number;
  damage_area_ratio: number;
  damage_mask_png: string;
  segmentation_method: string;
  priority_score: number;
  priority_level: PriorityLevel;
  components: {
    severity: number;
    damage_area: number;
    impact: number;
    impact_factor: number;
    weights: { severity: number; damage_area: number; impact: number };
  };
  formula: string;
  priority_rationale: string;
  assessment_timestamp: string;
  model_disagreement: {
    available: boolean;
    disagreement?: boolean;
    reason?: string;
    classifier_class?: string;
    unet_area_class?: string;
  };
  image_quality_warnings: string[];
  review_reasons: string[];
  uncertainty: number;
  human_verification: null | { status: string };
  stored: boolean;
  persistence_warning?: string;
  segmentation_model: string | null;
  model: string;
  notice: string;
};

const pretty = (label: string) => label.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

export default function RunPage() {
  const [postFile, setPostFile] = useState<File | null>(null);
  const [preFile, setPreFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prePreview, setPrePreview] = useState<string | null>(null);
  const [eventId, setEventId] = useState("");
  const [buildingUid, setBuildingUid] = useState("");
  const [coverageId, setCoverageId] = useState("");
  const [impactFactor, setImpactFactor] = useState("1");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [imageryAcquiredAt, setImageryAcquiredAt] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [apiStatus, setApiStatus] = useState<{ ready: boolean; warnings: string[]; status: string; persistence: string } | null>(null);

  useEffect(() => {
    let active = true;

    async function loadHealth() {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        const payload = await response.json();
        if (!active) return;
        setApiStatus({
          ready: Boolean(payload.classifier),
          warnings: Array.isArray(payload.warnings) ? payload.warnings : [],
          status: payload.status ?? "unknown",
          persistence: payload.persistence ?? "unknown",
        });
      } catch {
        if (!active) return;
        setApiStatus({ ready: false, warnings: ["Inference service is not reachable."], status: "offline", persistence: "unknown" });
      }
    }

    loadHealth();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
    if (prePreview) URL.revokeObjectURL(prePreview);
  }, [preview, prePreview]);

  function choosePost(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setPostFile(selected);
    setResult(null);
    setError(null);
    setPreview(selected ? URL.createObjectURL(selected) : null);
  }

  function choosePre(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setPreFile(selected);
    setPrePreview(selected ? URL.createObjectURL(selected) : null);
    setResult(null);
    setError(null);
  }

  async function analyze() {
    if (!postFile) return;
    setRunning(true);
    setError(null);
    setResult(null);
    const body = new FormData();
    const path = preFile ? "/api/analyze-pair" : "/api/analyze";
    if (preFile) {
      body.append("post_image", postFile);
      body.append("pre_image", preFile);
    } else {
      body.append("image", postFile);
    }
    if (eventId.trim()) body.append("event_id", eventId.trim());
    if (buildingUid.trim()) body.append("building_uid", buildingUid.trim());
    if (coverageId.trim()) body.append("coverage_id", coverageId.trim());
    if (latitude.trim()) body.append("latitude", latitude.trim());
    if (longitude.trim()) body.append("longitude", longitude.trim());
    if (imageryAcquiredAt) body.append("imagery_acquired_at", new Date(imageryAcquiredAt).toISOString());
    body.append("impact_factor", impactFactor);
    try {
      const response = await fetch(`${API_URL}${path}`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Analysis could not be completed.");
      setResult(data as Result);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to reach the inference service.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <section>
        <Badge variant="secondary" className="bg-orange-500/15 text-orange-200">
          Live model inference
        </Badge>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-white">Assess a building crop</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
          Upload a post-disaster building image. Add the pre-disaster image when you have it so the
          service can align the pair and estimate changed pixels. Results distinguish trained U-Net
          output from heuristic estimates. When PostGIS is connected, reports and image metadata are
          saved for review; otherwise the result explicitly indicates it was not saved.
        </p>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardTitle>Image input</CardTitle>
            <CardDescription>PNG, JPEG, or WebP up to 12 MB. Pre-disaster is optional.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <label className="flex min-h-64 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border border-dashed border-slate-600 bg-black/20 p-4 text-center hover:border-orange-400">
              {preview ? (
                // Local object URL selected by the user; Next image optimization is not applicable.
                // eslint-disable-next-line @next/next/no-img-element
                <img src={preview} alt="Selected building crop" className="max-h-60 rounded-lg object-contain" />
              ) : (
                <>
                  <Upload className="mb-3 size-8 text-orange-300" />
                  <span className="text-sm text-slate-300">Choose the post-disaster image</span>
                </>
              )}
              <input
                aria-label="Choose post-disaster building image"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={choosePost}
                className="sr-only"
              />
            </label>
            <label className="flex cursor-pointer items-center justify-between rounded-lg border border-white/10 px-3 py-2 text-sm text-slate-300 hover:border-orange-400">
              <span>{preFile ? preFile.name : "Optional pre-disaster image"}</span>
              <input
                aria-label="Choose pre-disaster image"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={choosePre}
                className="sr-only"
              />
            </label>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="space-y-1 text-xs text-slate-400">
                <span>Event ID (optional)</span>
                <input value={eventId} onChange={(event) => setEventId(event.target.value)} className="w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" />
              </label>
              <label className="space-y-1 text-xs text-slate-400">
                <span>Building ID (optional)</span>
                <input value={buildingUid} onChange={(event) => setBuildingUid(event.target.value)} className="w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" />
              </label>
              <label className="space-y-1 text-xs text-slate-400 sm:col-span-2">
                <span>Stable imagery coverage ID (optional; required for comparisons)</span>
                <input value={coverageId} onChange={(event) => setCoverageId(event.target.value)} className="w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" />
              </label>
              <label className="space-y-1 text-xs text-slate-400">
                <span>Latitude (optional)</span>
                <input type="number" min="-90" max="90" step="any" value={latitude} onChange={(event) => setLatitude(event.target.value)} className="w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" />
              </label>
              <label className="space-y-1 text-xs text-slate-400">
                <span>Longitude (optional)</span>
                <input type="number" min="-180" max="180" step="any" value={longitude} onChange={(event) => setLongitude(event.target.value)} className="w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" />
              </label>
              <label className="space-y-1 text-xs text-slate-400 sm:col-span-2">
                <span>Image acquisition time (optional; not the assessment time)</span>
                <input type="datetime-local" value={imageryAcquiredAt} onChange={(event) => setImageryAcquiredAt(event.target.value)} className="w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" />
              </label>
              <label className="space-y-1 text-xs text-slate-400 sm:col-span-2">
                <span>Contextual impact factor (0–1; neutral default 1.0; use documented context only)</span>
                <input type="number" min="0" max="1" step="0.05" value={impactFactor} onChange={(event) => setImpactFactor(event.target.value)} className="w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" />
              </label>
            </div>
            {postFile && (
              <p className="flex items-center gap-2 text-xs text-slate-400">
                <FileImage className="size-4" />
                {postFile.name} ({(postFile.size / 1024 / 1024).toFixed(1)} MB)
              </p>
            )}
            {apiStatus && (
              <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs text-slate-300">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium text-slate-200">API status</span>
                  <span className={apiStatus.ready ? "text-emerald-300" : "text-amber-300"}>{apiStatus.status}</span>
                </div>
                <p className="mt-1 text-slate-400">Persistence: {apiStatus.persistence}</p>
                {!apiStatus.ready && apiStatus.warnings.length > 0 && (
                  <ul className="mt-2 space-y-1 text-slate-400">
                    {apiStatus.warnings.map((warning) => (
                      <li key={warning}>• {warning}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            <Button className="w-full bg-orange-500 text-white hover:bg-orange-400" disabled={!postFile || running} onClick={analyze}>
              {running ? (
                <>
                  <LoaderCircle className="mr-2 size-4 animate-spin" />
                  Analyzing…
                </>
              ) : (
                "Run damage assessment"
              )}
            </Button>
            {error && (
              <p role="alert" className="rounded-md bg-red-500/10 p-3 text-sm text-red-200">
                {error}. Start the API service shown below and try again.
              </p>
            )}
          </CardContent>
        </Card>

        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardTitle>Assessment result</CardTitle>
            <CardDescription>{result ? `Model: ${result.model}` : "Your result will appear here."}</CardDescription>
          </CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-5">
                <div
                  className="rounded-xl border border-white/10 p-4"
                  style={{ borderLeftColor: COLORS[result.prediction], borderLeftWidth: 4 }}
                >
                  <p className="text-xs uppercase tracking-wide text-slate-500">AI predicted damage class</p>
                  <p className="mt-1 text-3xl font-semibold text-white">{pretty(result.prediction)}</p>
                  <p className="mt-2 text-sm text-slate-400">
                    Confidence {(result.confidence * 100).toFixed(1)}% · Severity {result.severity_score}/100 · Damaged
                    area {(result.damage_area_ratio * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="flex items-center justify-between rounded-lg bg-white/5 px-3 py-2">
                  <div>
                    <p className="text-xs text-slate-500">Priority score</p>
                    <p className="text-xl font-semibold tabular-nums text-white">{result.priority_score.toFixed(1)}</p>
                  </div>
                  <div className="rounded-lg border border-white/10 p-3 text-xs text-slate-400">
                    <p>{result.stored ? `Persisted assessment ${result.assessment_id} · ${result.review_status}` : result.persistence_warning ?? "This analysis was not persisted."}</p>
                    {result.assessment_id && (
                      <a className="mt-1 inline-block text-orange-300 hover:underline" href={`/history/${result.assessment_id}`}>
                        Open evidence report
                      </a>
                    )}
                    <p className="mt-1">Human-verified observation: {result.human_verification ? "available" : "not reviewed"}. AI output is not a verified observation.</p>
                  </div>
                  <div className="rounded-lg border border-white/10 p-3 text-xs text-slate-300">
                    <p className="font-medium text-slate-100">Priority rationale</p>
                    <p className="mt-1">{result.priority_rationale}</p>
                    <p className="mt-1 break-all text-slate-500">{result.formula}</p>
                    <p className="mt-2">
                      Components: severity {(result.components.severity * 100).toFixed(1)}% · area{" "}
                      {(result.components.damage_area * 100).toFixed(1)}% · impact factor{" "}
                      {result.components.impact_factor.toFixed(2)} (neutral default 1.0).
                      Confidence/uncertainty is separate and does not reduce this score.
                    </p>
                  </div>
                  <div className="space-y-2 rounded-lg border border-white/10 p-3 text-xs">
                    <p className="font-medium text-slate-100">Review signals</p>
                    <p className="text-slate-300">
                      Model comparison:{" "}
                      {result.model_disagreement.available
                        ? result.model_disagreement.disagreement
                          ? `Disagreement (${result.model_disagreement.classifier_class} vs ${result.model_disagreement.unet_area_class})`
                          : `No thresholded disagreement (${result.model_disagreement.classifier_class} vs ${result.model_disagreement.unet_area_class}); agreement does not prove correctness`
                        : `Unavailable — ${result.model_disagreement.reason}`}
                    </p>
                    <p className="text-slate-400">
                      Segmentation: {result.segmentation_method}
                      {result.segmentation_model ? ` (${result.segmentation_model})` : " — heuristic estimate, not trained U-Net output"}
                    </p>
                    {result.review_reasons.length > 0 && (
                      <p className="text-amber-200">Human review recommended: {result.review_reasons.join(", ")}</p>
                    )}
                    {result.image_quality_warnings.map((warning) => (
                      <p key={warning} className="text-amber-200">{warning}</p>
                    ))}
                    <p className="text-slate-500">
                      Assessment time: {new Date(result.assessment_timestamp).toLocaleString()} · Human verification: not reviewed
                    </p>
                  </div>
                  <Badge className={levelClass(result.priority_level)}>{result.priority_level}</Badge>
                </div>
                {(preview || prePreview) && (
                  <div className="grid grid-cols-2 gap-3">
                    {prePreview && (
                      <figure>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={prePreview} alt="Uploaded pre-disaster image" className="h-36 w-full rounded-lg object-contain bg-black/30" />
                        <figcaption className="mt-1 text-xs text-slate-500">Pre-disaster image</figcaption>
                      </figure>
                    )}
                    {preview && (
                    <figure>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={preview} alt="Uploaded crop" className="h-36 w-full rounded-lg object-contain bg-black/30" />
                      <figcaption className="mt-1 text-xs text-slate-500">Post-disaster image</figcaption>
                    </figure>
                    )}
                    <figure>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`data:image/png;base64,${result.damage_mask_png}`}
                        alt="Estimated damaged pixels"
                        className="h-36 w-full rounded-lg object-contain bg-black/30"
                      />
                      <figcaption className="mt-1 text-xs text-slate-500">
                        Mask ({result.segmentation_method})
                      </figcaption>
                    </figure>
                  </div>
                )}
                <div className="space-y-3">
                  {LABELS.map((label) => (
                    <div key={label}>
                      <div className="mb-1 flex justify-between text-sm text-slate-300">
                        <span>{pretty(label)}</span>
                        <span>{(result.probabilities[label] * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={result.probabilities[label] * 100} />
                    </div>
                  ))}
                </div>
                <p className="flex gap-2 text-xs leading-5 text-amber-100/80">
                  <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-300" />
                  {result.notice}
                </p>
              </div>
            ) : (
              <div className="flex min-h-64 flex-col items-center justify-center text-center text-sm text-slate-500">
                <FileImage className="mb-3 size-8" />
                Select an image and run the assessment.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="bg-[#0d1724] ring-white/10">
        <CardHeader>
          <CardTitle className="text-base">Start the inference service</CardTitle>
          <CardDescription>PostGIS enables persistent history and reviews. Without it, the API reports results as not saved. Trained ResNet50 weights are required for inference.</CardDescription>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-lg bg-black/40 p-4 text-xs text-slate-300">{`# PowerShell from the project root
docker compose up -d db
$env:DATABASE_URL = "postgresql+psycopg://damagemap:local-development-only@localhost:5432/damagemap"
uvicorn ml.api:app --host 0.0.0.0 --port 8000

# separate terminal
Set-Location web
npm run dev`}</pre>
        </CardContent>
      </Card>
    </div>
  );
}
