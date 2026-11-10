"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";

type Report = {
  id: string;
  created_at: string;
  imagery_acquired_at: string | null;
  event_id: string | null;
  building_uid: string | null;
  coverage_id: string | null;
  filename: string;
  prediction: string;
  confidence: number;
  uncertainty: number;
  probabilities: Record<string, number>;
  severity_score: number;
  damage_area_ratio: number;
  segmentation_method: string;
  segmentation_model_version: string | null;
  model_version: string;
  model_disagreement: { available: boolean; disagreement?: boolean; reason?: string; classifier_class?: string; unet_area_class?: string };
  decision_rules: {
    minimum_confidence_for_confirmation: number;
    minimum_class_gap: number;
    unet_area_class_thresholds: Record<string, number | string>;
  };
  image_quality_warnings: string[];
  review_reasons: string[];
  priority_score: number;
  priority_level: string;
  priority_components: { severity: number; damage_area: number; impact: number; impact_factor: number; weights: Record<string, number> };
  priority_formula: string;
  priority_rationale: string;
  review_status: string;
  human_damage_class: string | null;
  reviewer: string | null;
  review_notes: string | null;
  reviewed_at: string | null;
  post_image_metadata: { width: number; height: number; content_type: string; size_bytes: number };
  pre_image_metadata: { width: number; height: number; content_type: string; size_bytes: number } | null;
  images: { post: string; pre: string | null; mask: string };
};

export default function AssessmentReportPage() {
  const { assessment_id: assessmentId } = useParams<{ assessment_id: string }>();
  const [report, setReport] = useState<Report | null>(null);
  const [comparison, setComparison] = useState<{ comparable: boolean; reason?: string; conclusion?: string; previous?: { imagery_acquired_at: string; prediction: string; damage_area_ratio: number }; current?: { imagery_acquired_at: string; prediction: string; damage_area_ratio: number } } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/api/assessments/${assessmentId}`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail ?? "Report could not be loaded.");
        if (active) setReport(data);
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Could not reach the report service.");
      });
    fetch(`${API_URL}/api/assessments/${assessmentId}/comparisons`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail ?? "Comparison could not be loaded.");
        if (active) setComparison(data);
      })
      .catch(() => {
        if (active) setComparison({ comparable: false, reason: "Comparison information is unavailable." });
      });
    return () => { active = false; };
  }, [assessmentId]);

  if (error) return <p role="alert" className="text-amber-200">{error}</p>;
  if (!report) return <p className="text-slate-400">Loading evidence report…</p>;

  return (
    <div className="space-y-5">
      <section className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Badge variant="secondary">Evidence report · AI output is not verified observation</Badge>
          <h1 className="mt-3 text-2xl font-semibold text-white">{report.filename}</h1>
          <p className="mt-1 text-sm text-slate-400">{report.event_id ?? "No event"} · {report.building_uid ?? "No building ID"} · Coverage {report.coverage_id ?? "not supplied"} · Assessed {new Date(report.created_at).toLocaleString()}</p>
        </div>
        <Badge>{report.review_status}</Badge>
      </section>

      <Card className="bg-[#0d1724] ring-white/10">
        <CardHeader><CardTitle>Imagery evidence</CardTitle></CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-3">
          {report.pre_image_metadata && report.images.pre && <ImageEvidence title="Pre-disaster" src={`${API_URL}${report.images.pre}`} metadata={report.pre_image_metadata} />}
          <ImageEvidence title="Post-disaster" src={`${API_URL}${report.images.post}`} metadata={report.post_image_metadata} />
          <ImageEvidence title={`Damage mask · ${report.segmentation_method}`} src={`${API_URL}${report.images.mask}`} metadata={null} />
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader><CardTitle>AI prediction (not a verified observation)</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-300">
            <p>Damage class: <strong>{report.prediction}</strong> · confidence {(report.confidence * 100).toFixed(1)}% · uncertainty {(report.uncertainty * 100).toFixed(1)}%</p>
            <p>Ordinal severity: {report.severity_score.toFixed(1)}/100 · estimated affected area: {(report.damage_area_ratio * 100).toFixed(1)}%</p>
            <p>Classifier: {report.model_version} · segmentation: {report.segmentation_model_version ?? "No trained U-Net; heuristic estimate"}</p>
            <div className="pt-2">{Object.entries(report.probabilities).map(([label, probability]) => <p key={label}>{label}: {(probability * 100).toFixed(1)}%</p>)}</div>
            <p>U-Net comparison: {report.model_disagreement.available ? report.model_disagreement.disagreement ? `Disagreement (${report.model_disagreement.classifier_class} vs ${report.model_disagreement.unet_area_class})` : "No thresholded disagreement; agreement does not prove correctness" : `Unavailable — ${report.model_disagreement.reason}`}</p>
            {report.review_reasons.map((reason) => <p key={reason} className="text-amber-200">Review reason: {reason}</p>)}
            {report.image_quality_warnings.map((warning) => <p key={warning} className="text-amber-200">{warning}</p>)}
            <p className="text-xs text-slate-500">Stored comparison rules: class gap ≥ {report.decision_rules.minimum_class_gap}; confidence review threshold {(report.decision_rules.minimum_confidence_for_confirmation * 100).toFixed(0)}%; U-Net area bins {JSON.stringify(report.decision_rules.unet_area_class_thresholds)}.</p>
          </CardContent>
        </Card>
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader><CardTitle>Priority and verification</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-300">
            <p>{report.priority_score.toFixed(2)}/100 · {report.priority_rationale}</p>
            <p>Components: severity {report.priority_components.severity.toFixed(3)}, area {report.priority_components.damage_area.toFixed(3)}, impact {report.priority_components.impact.toFixed(3)} (factor {report.priority_components.impact_factor.toFixed(2)})</p>
            <p>Weights: {Object.entries(report.priority_components.weights).map(([key, value]) => `${key} ${value}`).join(" · ")}</p>
            <p className="break-all text-xs text-slate-500">Calculation: {report.priority_formula}</p>
            <p>Confidence is separate and does not lower priority. This is not an official emergency standard.</p>
            <hr className="border-white/10" />
            <p>Human verification: {report.review_status === "confirmed" ? `Confirmed: ${report.human_damage_class}` : report.review_status === "corrected" ? `Corrected to ${report.human_damage_class}` : report.review_status}</p>
            {report.reviewer && <p>Reviewer: {report.reviewer} · {report.reviewed_at ? new Date(report.reviewed_at).toLocaleString() : ""}</p>}
            {report.review_notes && <p>Review notes: {report.review_notes}</p>}
            <p>Imagery acquisition: {report.imagery_acquired_at ? new Date(report.imagery_acquired_at).toLocaleString() : "not supplied; assessment timestamp is not an acquisition timestamp"}</p>
          </CardContent>
        </Card>
      </div>
      <Card className="bg-[#0d1724] ring-white/10">
        <CardHeader><CardTitle>Successive assessment comparison</CardTitle></CardHeader>
        <CardContent className="space-y-2 text-sm text-slate-300">
          {!comparison ? <p>Checking for a comparable earlier assessment…</p> : !comparison.comparable ? <p>{comparison.reason}</p> : (
            <>
              <p>Prior output: {comparison.previous?.prediction} · {(comparison.previous!.damage_area_ratio * 100).toFixed(1)}% estimated area · acquired {new Date(comparison.previous!.imagery_acquired_at).toLocaleString()}</p>
              <p>Current output: {comparison.current?.prediction} · {(comparison.current!.damage_area_ratio * 100).toFixed(1)}% estimated area · acquired {new Date(comparison.current!.imagery_acquired_at).toLocaleString()}</p>
              <p>{comparison.conclusion}</p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ImageEvidence({ title, src, metadata }: { title: string; src: string; metadata: { width: number; height: number; content_type: string; size_bytes: number } | null }) {
  return (
    <figure>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={title} className="h-52 w-full rounded-lg bg-black/30 object-contain" />
      <figcaption className="mt-2 text-sm text-slate-300">{title}</figcaption>
      {metadata && <p className="text-xs text-slate-500">{metadata.width} × {metadata.height} · {metadata.content_type} · {(metadata.size_bytes / 1024).toFixed(0)} KB</p>}
    </figure>
  );
}
