"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useCatalog } from "@/hooks/use-catalog";
import { levelClass, priorityScore, zoneDamageArea, type PriorityLevel } from "@/lib/scoring";
import Link from "next/link";
import { useMemo, useState } from "react";

const TYPES = ["all", "flood", "fire", "tsunami", "earthquake"] as const;

export default function PriorityPage() {
  const { catalog, error, loading } = useCatalog();
  const [kind, setKind] = useState<(typeof TYPES)[number]>("all");

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
        const scored = priorityScore(event.severity_pred, zoneDamageArea(labels), confidence);
        return { event, confidence, ...scored };
      })
      .sort((a, b) => b.priority_score - a.priority_score);
  }, [catalog, kind]);

  if (loading) return <p className="text-slate-400">Ranking priority zones…</p>;
  if (error || !catalog) return <p className="text-red-300">{error ?? "Catalog missing."}</p>;

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <Badge variant="secondary" className="bg-orange-500/15 text-orange-200">
          In-memory ranking
        </Badge>
        <h1 className="text-3xl font-semibold tracking-tight text-white">Priority zones</h1>
        <p className="max-w-2xl text-sm leading-6 text-slate-400">
          Each tile is scored as 0.5 × severity + 0.3 × damaged-area share + 0.2 × severity ×
          confidence. The list is computed in the browser from the demo catalog. The same ranking
          is available from <span className="text-slate-300">GET /api/priority-zones</span> and is
          not written to a database.
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
                Severity {zone.event.severity_pred.toFixed(1)} · confidence {(zone.confidence * 100).toFixed(0)}%
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
