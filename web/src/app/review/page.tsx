"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";
const CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"] as const;

type ReviewItem = {
  id: string;
  filename: string;
  created_at: string;
  event_id: string | null;
  building_uid: string | null;
  prediction: string;
  confidence: number;
  uncertainty: number;
  review_reasons: string[];
  model_disagreement: { available: boolean; disagreement?: boolean; reason?: string; unet_area_class?: string };
};

export default function ReviewPage() {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [reviewer, setReviewer] = useState("");
  const [corrections, setCorrections] = useState<Record<string, (typeof CLASSES)[number]>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchReviewQueue()
      .then((queue) => {
        if (active) {
          setItems(queue);
          setError(null);
        }
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Could not reach the review service.");
      });
    return () => { active = false; };
  }, []);

  async function submitReview(item: ReviewItem, status: "confirmed" | "corrected" | "unresolved") {
    const reviewerName = reviewer.trim();
    if (!reviewerName) {
      setError("Enter a reviewer name before submitting a decision.");
      return;
    }
    setBusy(item.id);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/api/assessments/${item.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          status,
          reviewer: reviewerName,
          ...(status === "corrected" ? { corrected_class: corrections[item.id] ?? item.prediction } : {}),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Review could not be saved.");
      setItems(await fetchReviewQueue());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save this review.");
    } finally {
      setBusy(null);
    }

  }

  return (
    <div className="space-y-6">
      <section>
        <Badge variant="secondary" className="bg-amber-500/15 text-amber-200">Human verification</Badge>
        <h1 className="mt-3 text-3xl font-semibold text-white">Review queue</h1>
        <p className="mt-2 text-sm text-slate-400">Items are queued for low confidence, model disagreement, or image-quality warnings. Human corrections are stored separately and are not automatically used for training.</p>
      </section>
      <label className="block max-w-sm text-xs text-slate-400">Reviewer name (prototype field; not authenticated)<input value={reviewer} onChange={(event) => setReviewer(event.target.value)} className="mt-1 w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" /></label>
      {error && <p role="alert" className="rounded bg-amber-500/10 p-3 text-sm text-amber-100">{error} Ensure PostGIS and the API are running.</p>}
      {!error && items.length === 0 && <p className="text-sm text-slate-500">No assessments are waiting for review.</p>}
      <div className="grid gap-4">
        {items.map((item) => (
          <Card key={item.id} className="bg-[#0d1724] ring-white/10">
            <CardHeader className="flex-row items-start justify-between">
              <div><CardTitle className="text-base">{item.filename} · AI: {item.prediction}</CardTitle><p className="mt-1 text-xs text-slate-400">{item.event_id ?? "No event"} · {item.building_uid ?? "No building ID"} · {new Date(item.created_at).toLocaleString()}</p></div>
              <Badge variant="secondary">Pending</Badge>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-slate-300">Confidence {(item.confidence * 100).toFixed(1)}% · Uncertainty {(item.uncertainty * 100).toFixed(1)}% · Reasons: {item.review_reasons.join(", ")}</p>
              <p className="text-xs text-slate-400">Model comparison: {item.model_disagreement.available ? item.model_disagreement.disagreement ? `Disagreement with U-Net area class ${item.model_disagreement.unet_area_class}` : "No thresholded disagreement (not proof of correctness)" : `Unavailable — ${item.model_disagreement.reason}`}</p>
              <div className="flex flex-wrap items-center gap-2">
                <select aria-label={`Correction for ${item.filename}`} value={corrections[item.id] ?? item.prediction as (typeof CLASSES)[number]} onChange={(event) => setCorrections((current) => ({ ...current, [item.id]: event.target.value as (typeof CLASSES)[number] }))} className="rounded border border-white/10 bg-[#0d1724] px-3 py-2 text-sm text-white">
                  {CLASSES.map((damageClass) => <option key={damageClass} value={damageClass}>{damageClass}</option>)}
                </select>
                <Button disabled={busy === item.id} onClick={() => submitReview(item, "confirmed")}>Confirm</Button>
                <Button variant="outline" disabled={busy === item.id} onClick={() => submitReview(item, "corrected")}>Correct</Button>
                <Button variant="outline" disabled={busy === item.id} onClick={() => submitReview(item, "unresolved")}>Unresolved</Button>
                <Button asChild size="sm" variant="ghost"><Link href={`/history/${item.id}`}>Evidence report</Link></Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

async function fetchReviewQueue(): Promise<ReviewItem[]> {
  const response = await fetch(`${API_URL}/api/reviews`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail ?? "Review queue could not be loaded.");
  return data.assessments;
}
