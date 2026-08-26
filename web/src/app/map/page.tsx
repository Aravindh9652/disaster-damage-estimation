"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useCatalog } from "@/hooks/use-catalog";
import {
  CLASS_ORDER,
  prettyClass,
  severityBand,
  type Building,
  type DamageClass,
  type DisasterEvent,
} from "@/lib/catalog";
import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

const COLORS: Record<DamageClass, string> = {
  "no-damage": "#22c55e",
  "minor-damage": "#eab308",
  "major-damage": "#f97316",
  destroyed: "#ef4444",
};

function MapInner() {
  const { catalog, error, loading } = useCatalog();
  const params = useSearchParams();
  const [eventId, setEventId] = useState<string | null>(params.get("event"));
  const [selected, setSelected] = useState<Building | null>(null);
  const [showPred, setShowPred] = useState(true);

  const event: DisasterEvent | undefined = useMemo(() => {
    if (!catalog) return undefined;
    return (
      catalog.events.find((e) => e.id === eventId) ?? catalog.events[0]
    );
  }, [catalog, eventId]);

  if (loading) return <p className="text-slate-400">Loading tiles…</p>;
  if (error || !catalog || !event) {
    return <p className="text-red-300">{error ?? "No events"}</p>;
  }

  const colorOf = (b: Building) => COLORS[showPred ? b.pred : b.label];
  const band = severityBand(event.severity_pred);

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white">Damage map</h1>
          <p className="text-sm text-slate-400">
            Polygons from building extraction, colored by ResNet50 class.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {catalog.events.map((e) => (
            <Button
              key={e.id}
              size="sm"
              variant={e.id === event.id ? "default" : "outline"}
              onClick={() => {
                setEventId(e.id);
                setSelected(null);
              }}
            >
              {e.name}
            </Button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant={showPred ? "secondary" : "outline"}
          onClick={() => setShowPred(true)}
        >
          Predicted
        </Button>
        <Button
          size="sm"
          variant={!showPred ? "secondary" : "outline"}
          onClick={() => setShowPred(false)}
        >
          Ground truth
        </Button>
        {CLASS_ORDER.map((id) => (
          <span key={id} className="flex items-center gap-1.5 text-xs text-slate-400">
            <span className="size-2.5 rounded-full" style={{ background: COLORS[id] }} />
            {prettyClass(id)}
          </span>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(280px,0.9fr)]">
        <Card className="overflow-hidden bg-[#0d1724] p-0 ring-white/10">
          <div className="relative">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={event.tile}
              alt={event.name}
              className="block w-full"
            />
            <svg
              viewBox={`0 0 ${event.width} ${event.height}`}
              className="absolute inset-0 h-full w-full"
            >
              {event.buildings.map((b) => {
                const points = b.polygon.map((p) => `${p.x},${p.y}`).join(" ");
                const active = selected?.uid === b.uid;
                return (
                  <polygon
                    key={b.uid}
                    points={points}
                    fill={colorOf(b)}
                    fillOpacity={active ? 0.55 : 0.28}
                    stroke={colorOf(b)}
                    strokeWidth={active ? 3 : 1.5}
                    className="cursor-pointer"
                    onClick={() => setSelected(b)}
                  />
                );
              })}
            </svg>
          </div>
        </Card>

        <div className="space-y-4">
          <Card className="bg-[#0d1724] ring-white/10">
            <CardHeader>
              <div className="flex items-start justify-between gap-2">
                <div>
                  <CardTitle>{event.name}</CardTitle>
                  <CardDescription>
                    {event.place} · {event.buildings.length} buildings
                  </CardDescription>
                </div>
                <Badge className={band.className}>{band.label}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <Row label="Severity (pred)" value={event.severity_pred.toFixed(1)} />
              <Row label="Severity (labeled)" value={event.severity_true.toFixed(1)} />
              {CLASS_ORDER.map((id) => (
                <Row
                  key={id}
                  label={prettyClass(id)}
                  value={String(event.counts_pred[id])}
                />
              ))}
            </CardContent>
          </Card>

          {selected ? (
            <Card className="bg-[#0d1724] ring-white/10">
              <CardHeader>
                <CardTitle className="text-sm">{selected.uid}</CardTitle>
                <CardDescription>
                  pred {prettyClass(selected.pred)} · true {prettyClass(selected.label)}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={selected.crop}
                  alt={selected.uid}
                  className="w-full rounded-md ring-1 ring-white/10"
                />
                {CLASS_ORDER.map((id) => (
                  <div key={id}>
                    <div className="mb-1 flex justify-between text-xs text-slate-400">
                      <span>{prettyClass(id)}</span>
                      <span>{(selected.probs[id] * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${selected.probs[id] * 100}%`,
                          background: COLORS[id],
                        }}
                      />
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          ) : (
            <p className="text-sm text-slate-500">
              Click a building polygon to inspect the extracted crop and class probabilities.
            </p>
          )}

          <Card className="bg-[#0d1724] ring-white/10">
            <CardHeader>
              <CardTitle className="text-sm">Buildings</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-48">
                <ul className="space-y-1">
                  {event.buildings.map((b) => (
                    <li key={b.uid}>
                      <button
                        type="button"
                        onClick={() => setSelected(b)}
                        className="flex w-full items-center justify-between rounded px-2 py-1 text-left text-xs hover:bg-white/5"
                      >
                        <span className="font-mono text-slate-300">{b.uid}</span>
                        <span style={{ color: COLORS[b.pred] }}>
                          {prettyClass(b.pred)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-slate-300">
      <span className="text-slate-500">{label}</span>
      <span className="tabular-nums">{value}</span>
    </div>
  );
}

export default function MapPage() {
  return (
    <Suspense fallback={<p className="text-slate-400">Loading map…</p>}>
      <MapInner />
    </Suspense>
  );
}
