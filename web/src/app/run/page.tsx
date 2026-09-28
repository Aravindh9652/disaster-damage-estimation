"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, FileImage, LoaderCircle, Upload } from "lucide-react";
import { useState } from "react";
import type { ChangeEvent } from "react";

const API_URL = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";
const LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"] as const;
const COLORS: Record<(typeof LABELS)[number], string> = {
  "no-damage": "#22c55e", "minor-damage": "#eab308", "major-damage": "#f97316", destroyed: "#ef4444",
};
type Result = { filename: string; prediction: (typeof LABELS)[number]; confidence: number; probabilities: Record<(typeof LABELS)[number], number>; severity_score: number; model: string; notice: string };
const pretty = (label: string) => label.replace("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

export default function RunPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected); setResult(null); setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(selected ? URL.createObjectURL(selected) : null);
  }
  async function analyze() {
    if (!file) return;
    setRunning(true); setError(null); setResult(null);
    const body = new FormData(); body.append("image", file);
    try {
      const response = await fetch(`${API_URL}/api/analyze`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Analysis could not be completed.");
      setResult(data as Result);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Unable to reach the inference service."); }
    finally { setRunning(false); }
  }
  return <div className="mx-auto max-w-5xl space-y-6">
    <section><Badge variant="secondary" className="bg-orange-500/15 text-orange-200">Live model inference</Badge>
      <h1 className="mt-3 text-3xl font-semibold tracking-tight text-white">Assess a building crop</h1>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Upload one post-disaster building image. The portal sends it to the local ResNet50 checkpoint and returns its four-level xBD damage estimate with confidence and a transparent severity score.</p>
    </section>
    <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
      <Card className="bg-[#0d1724] ring-white/10"><CardHeader><CardTitle>Image input</CardTitle><CardDescription>PNG, JPEG, or WebP up to 12 MB.</CardDescription></CardHeader><CardContent className="space-y-4">
        <label className="flex min-h-64 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border border-dashed border-slate-600 bg-black/20 p-4 text-center hover:border-orange-400">
          {preview ? <>
            {/* Local object URL selected by the user; Next image optimization is not applicable. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="Selected building crop" className="max-h-60 rounded-lg object-contain" />
          </> : <><Upload className="mb-3 size-8 text-orange-300" /><span className="text-sm text-slate-300">Choose a building image</span></>}
          <input aria-label="Choose building image" type="file" accept="image/png,image/jpeg,image/webp" onChange={chooseFile} className="sr-only" />
        </label>
        {file && <p className="flex items-center gap-2 text-xs text-slate-400"><FileImage className="size-4" />{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</p>}
        <Button className="w-full bg-orange-500 text-white hover:bg-orange-400" disabled={!file || running} onClick={analyze}>{running ? <><LoaderCircle className="mr-2 size-4 animate-spin" />Analyzing with ResNet50…</> : "Run damage assessment"}</Button>
        {error && <p role="alert" className="rounded-md bg-red-500/10 p-3 text-sm text-red-200">{error}. Start the API service shown below and try again.</p>}
      </CardContent></Card>
      <Card className="bg-[#0d1724] ring-white/10"><CardHeader><CardTitle>Assessment result</CardTitle><CardDescription>{result ? `Model: ${result.model}` : "Your result will appear here."}</CardDescription></CardHeader><CardContent>
        {result ? <div className="space-y-5"><div className="rounded-xl border border-white/10 p-4" style={{ borderLeftColor: COLORS[result.prediction], borderLeftWidth: 4 }}><p className="text-xs uppercase tracking-wide text-slate-500">Predicted damage class</p><p className="mt-1 text-3xl font-semibold text-white">{pretty(result.prediction)}</p><p className="mt-2 text-sm text-slate-400">Confidence {(result.confidence * 100).toFixed(1)}% · Severity score {result.severity_score}/100</p></div>
          <div className="space-y-3">{LABELS.map((label) => <div key={label}><div className="mb-1 flex justify-between text-sm text-slate-300"><span>{pretty(label)}</span><span>{(result.probabilities[label] * 100).toFixed(1)}%</span></div><Progress value={result.probabilities[label] * 100} /></div>)}</div>
          <p className="flex gap-2 text-xs leading-5 text-amber-100/80"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-300" />{result.notice}</p></div> : <div className="flex min-h-64 flex-col items-center justify-center text-center text-sm text-slate-500"><FileImage className="mb-3 size-8" />Select an image and run the assessment.</div>}
      </CardContent></Card>
    </div>
    <Card className="bg-[#0d1724] ring-white/10"><CardHeader><CardTitle className="text-base">Start the inference service</CardTitle><CardDescription>The Next.js portal and PyTorch model run as separate local services.</CardDescription></CardHeader><CardContent><pre className="overflow-x-auto rounded-lg bg-black/40 p-4 text-xs text-slate-300">{`# from the project root, after installing ml/requirements.txt\nuvicorn ml.api:app --host 0.0.0.0 --port 8000\n\n# separate terminal\ncd web\nnpm run dev`}</pre></CardContent></Card>
  </div>;
}
