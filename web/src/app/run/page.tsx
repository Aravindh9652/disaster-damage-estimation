"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { levelClass, type PriorityLevel } from "@/lib/scoring";
import { AlertTriangle, FileImage, LoaderCircle, Upload } from "lucide-react";
import { useState } from "react";
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
  model: string;
  notice: string;
};

const pretty = (label: string) => label.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

export default function RunPage() {
  const [postFile, setPostFile] = useState<File | null>(null);
  const [preFile, setPreFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  function choosePost(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setPostFile(selected);
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(selected ? URL.createObjectURL(selected) : null);
  }

  function choosePre(event: ChangeEvent<HTMLInputElement>) {
    setPreFile(event.target.files?.[0] ?? null);
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
          service can align the pair and mark changed pixels. The response includes the Joint Damage
          Scale class, a damage-area estimate, and a priority score. Nothing is saved.
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
            {postFile && (
              <p className="flex items-center gap-2 text-xs text-slate-400">
                <FileImage className="size-4" />
                {postFile.name} ({(postFile.size / 1024 / 1024).toFixed(1)} MB)
              </p>
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
                  <p className="text-xs uppercase tracking-wide text-slate-500">Predicted damage class</p>
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
                  <Badge className={levelClass(result.priority_level)}>{result.priority_level}</Badge>
                </div>
                {preview && (
                  <div className="grid grid-cols-2 gap-3">
                    <figure>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={preview} alt="Uploaded crop" className="h-36 w-full rounded-lg object-contain bg-black/30" />
                      <figcaption className="mt-1 text-xs text-slate-500">Uploaded crop</figcaption>
                    </figure>
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
          <CardDescription>The Next.js portal and PyTorch model run as separate local services. Results are not stored.</CardDescription>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-lg bg-black/40 p-4 text-xs text-slate-300">{`# from the project root, after installing ml/requirements.txt
uvicorn ml.api:app --host 0.0.0.0 --port 8000

# separate terminal
cd web
npm run dev`}</pre>
        </CardContent>
      </Card>
    </div>
  );
}
