"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";
const CLASSES = ["", "no-damage", "minor-damage", "major-damage", "destroyed"] as const;
const STATUSES = ["", "unreviewed", "pending", "confirmed", "corrected", "unresolved"] as const;

type Assessment = {
  id: string;
  created_at: string;
  event_id: string | null;
  building_uid: string | null;
  filename: string;
  prediction: string;
  confidence: number;
  uncertainty: number;
  severity_score: number;
  priority_score: number;
  review_status: string;
  human_damage_class: string | null;
  human_verification: { status: string; damage_class: string | null; reviewer: string; reviewed_at: string } | null;
};

export default function HistoryPage() {
  const [eventId, setEventId] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [severity, setSeverity] = useState<(typeof CLASSES)[number]>("");
  const [status, setStatus] = useState<(typeof STATUSES)[number]>("");
  const [assessments, setAssessments] = useState<Assessment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function loadHistory(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    const query = new URLSearchParams();
    if (eventId.trim()) query.set("event_id", eventId.trim());
    if (dateFrom) query.set("date_from", dateFrom);
    if (dateTo) query.set("date_to", dateTo);
    if (severity) query.set("severity", severity);
    if (status) query.set("review_status", status);
    try {
      const response = await fetch(`${API_URL}/api/assessments?${query}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "History could not be loaded.");
      setAssessments(data.assessments);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not reach the history service.");
      setAssessments([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <Badge variant="secondary" className="bg-orange-500/15 text-orange-200">Persisted assessments</Badge>
        <h1 className="mt-3 text-3xl font-semibold text-white">Assessment history</h1>
        <p className="mt-2 text-sm text-slate-400">
          Only real saved inference records appear here. Demo catalog rows are not presented as saved analyses.
        </p>
      </section>
      <Card className="bg-[#0d1724] ring-white/10">
        <CardHeader><CardTitle className="text-base">Filters</CardTitle></CardHeader>
        <CardContent>
          <form onSubmit={loadHistory} className="grid gap-3 md:grid-cols-3">
            <input aria-label="Event ID" placeholder="Event ID" value={eventId} onChange={(event) => setEventId(event.target.value)} className="rounded border border-white/10 bg-black/20 px-3 py-2 text-sm" />
            <label className="text-xs text-slate-400">From<input type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} className="mt-1 block w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" /></label>
            <label className="text-xs text-slate-400">To<input type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} className="mt-1 block w-full rounded border border-white/10 bg-black/20 px-3 py-2 text-sm text-white" /></label>
            <select aria-label="AI predicted severity" value={severity} onChange={(event) => setSeverity(event.target.value as (typeof CLASSES)[number])} className="rounded border border-white/10 bg-[#0d1724] px-3 py-2 text-sm text-white">
              {CLASSES.map((item) => <option key={item} value={item}>{item || "All severity"}</option>)}
            </select>
            <select aria-label="Verification status" value={status} onChange={(event) => setStatus(event.target.value as (typeof STATUSES)[number])} className="rounded border border-white/10 bg-[#0d1724] px-3 py-2 text-sm text-white">
              {STATUSES.map((item) => <option key={item} value={item}>{item || "All verification statuses"}</option>)}
            </select>
            <Button type="submit" disabled={loading}>{loading ? "Loading…" : "Apply filters"}</Button>
          </form>
        </CardContent>
      </Card>
      {error && <p role="alert" className="rounded bg-amber-500/10 p-3 text-sm text-amber-100">{error} Start PostGIS and the API to view persisted records.</p>}
      {!error && assessments.length === 0 && <p className="text-sm text-slate-500">Choose filters and apply to load history.</p>}
      <div className="grid gap-3">
        {assessments.map((item) => (
          <Card key={item.id} className="bg-[#0d1724] ring-white/10">
            <CardContent className="flex flex-wrap items-center justify-between gap-4 p-4">
              <div>
                <p className="font-medium text-white">{item.filename} · AI: {item.prediction}</p>
                <p className="mt-1 text-xs text-slate-400">
                  {new Date(item.created_at).toLocaleString()} · Event {item.event_id ?? "—"} · Building {item.building_uid ?? "—"}
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  Confidence {(item.confidence * 100).toFixed(1)}% · Uncertainty {(item.uncertainty * 100).toFixed(1)}% · Priority {item.priority_score.toFixed(1)} · {item.human_verification ? `Verified: ${item.human_damage_class ?? item.review_status}` : "Not human-verified"}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <Badge variant="secondary">{item.review_status}</Badge>
                <Button asChild size="sm" variant="outline"><Link href={`/history/${item.id}`}>Evidence report</Link></Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
