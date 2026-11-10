"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useCatalog } from "@/hooks/use-catalog";
import {
  DEFAULT_PRIORITY_CONFIGURATION,
  levelClass,
  priorityScore,
  zoneDamageArea,
  type PriorityConfiguration,
  type PriorityLevel,
} from "@/lib/scoring";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

const TYPES = ["all", "flood", "fire", "tsunami", "earthquake"] as const;
const API_URL = process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";

export default function PriorityPage() {
  const { catalog, error, loading } = useCatalog();
  const [kind, setKind] = useState<(typeof TYPES)[number]>("all");
  const [configuration, setConfiguration] = useState<PriorityConfiguration>(DEFAULT_PRIORITY_CONFIGURATION);
  const [configurationWarning, setConfigurationWarning] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/api/health`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail ?? "Scoring configuration is unavailable.");
        if (active && data.priority_configuration) {
          setConfiguration({
            weights: data.priority_configuration.weights,
            cutoffs: data.priority_configuration.cutoffs,
          });
          setConfigurationWarning(null);
        }
      })
      .catch(() => {
        if (active) setConfigurationWarning("Inference API unavailable; displaying documented default scoring settings.");
      });
    return () => { active = false; };
  }, []);

  const zones = useMemo(() => {
    if (!catalog) return [];
    return catalog.events
      .filter((event) => kind === "all" || event.type === kind)
      .map((event) => {
        const labels = event.buildings.map((building) => building.pred);
        const confidence =
          event.buildings.reduce((sum, building) => {
            const top = Math.max(...Object.values(building.probs));
            return sum + top;
          }, 0) / Math.max(event.buildings.length, 1);
        const scored = priorityScore(event.severity_pred, zoneDamageArea(labels), 1, configuration);
        return { event, confidence, ...scored };
      })
      .sort((a, b) => b.priority_score - a.priority_score);
  }, [catalog, configuration, kind]);

  if (loading) return <p className="text-slate-400">Ranking priority zones…</p>;
  if (error || !catalog) return <p className="text-red-300">{error ?? "Catalog missing."}</p>;

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <Badge variant="secondary" className="bg-orange-500/15 text-orange-200">
          Synthetic demo ranking
        </Badge>
        <h1 className="text-3xl font-semibold tracking-tight text-white">Priority zones</h1>
        <p className="max-w-2xl text-sm leading-6 text-slate-400">
          Project-defined score applied to synthetic catalog values: {configuration.weights.severity} × normalized severity + {configuration.weights.damage_area} × normalized
          damaged-area share + {configuration.weights.impact} × normalized severity × impact factor (neutral default 1.0).
          Confidence is shown separately as uncertainty and does not lower priority. The list uses
          demo catalog data; this is not an official emergency standard.
        </p>
        <p className="text-xs text-slate-500">
          Cutoffs: critical ≥ {configuration.cutoffs.critical}, high ≥ {configuration.cutoffs.high}, moderate ≥ {configuration.cutoffs.moderate}; lower scores are low.
          {configurationWarning && ` ${configurationWarning}`}
        </p>
      </section>

      <div className="flex flex-wrap gap-2">
        {TYPES.map((type) => (
          <Button
            key={type}
            size="sm"
            variant={kind === type ? "default" : "outline"}
            onClick={() => setKind(type)}
          >
            {type}
          </Button>
        ))}
      </div>

      <div className="grid gap-4">
        {zones.map((zone, index) => (
          <Card key={zone.event.id} className="bg-[#0d1724] ring-white/10">
            <CardHeader className="flex-row items-start justify-between gap-3">
              <div>
                <CardDescription>Rank {index + 1}</CardDescription>
                <CardTitle>{zone.event.name}</CardTitle>
                <p className="mt-1 text-sm text-slate-400">
                  {zone.event.place} · {zone.event.type} · {zone.event.buildings.length} buildings
                </p>
              </div>
              <Badge className={levelClass(zone.priority_level as PriorityLevel)}>
                {zone.priority_level}
              </Badge>
            </CardHeader>
            <CardContent className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <p className="text-xs text-slate-500">Priority score</p>
                <p className="text-3xl font-semibold tabular-nums text-white">
                  {zone.priority_score.toFixed(1)}
                </p>
              </div>
              <p className="text-sm text-slate-400">
                Severity {zone.event.severity_pred.toFixed(1)} · uncertainty {((1 - zone.confidence) * 100).toFixed(0)}%
              </p>
              <Button asChild size="sm" variant="outline">
                <Link href={`/map?event=${zone.event.id}`}>Open map</Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
