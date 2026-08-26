"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useCatalog } from "@/hooks/use-catalog";
import { prettyClass, severityBand } from "@/lib/catalog";
import { CheckCircle2, Layers3, TriangleAlert } from "lucide-react";
import Link from "next/link";

export default function OverviewPage() {
  const { catalog, error, loading } = useCatalog();

  if (loading) {
    return <p className="text-slate-400">Loading extracted buildings and model scores…</p>;
  }
  if (error || !catalog) {
    return (
      <p className="text-red-300">
        {error ?? "Catalog missing. Run python ml/generate_demo_artifacts.py"}
      </p>
    );
  }

  const { metrics, events, pipeline, classes } = catalog;
  const hardest = Object.entries(metrics.per_class_f1).sort((a, b) => a[1] - b[1])[0];

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <Badge variant="secondary" className="bg-orange-500/15 text-orange-200">
          Continuation after building extraction
        </Badge>
        <h1 className="max-w-3xl text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          Classify every extracted building, then rank tiles by severity.
        </h1>
        <p className="max-w-2xl text-slate-400">
          Your xBD work already produced post-disaster crops. This slice trains a
          ResNet50 Joint Damage Scale head, reports weighted F1 (not accuracy),
          and rolls predictions into a 0–100 neighborhood severity index.
        </p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi label="Buildings classified" value={String(metrics.n_buildings)} hint="post-disaster crops" />
        <Kpi
          label="Weighted F1"
          value={metrics.weighted_f1.toFixed(3)}
          hint="xView2 primary metric"
        />
        <Kpi
          label="Hardest class"
          value={prettyClass(hardest[0])}
          hint={`F1 ${hardest[1].toFixed(2)}`}
        />
        <Kpi
          label="Events mapped"
          value={String(metrics.n_events)}
          hint="flood, fire, tsunami, quake"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardTitle>Pipeline from your notebook</CardTitle>
            <CardDescription>
              Steps 1–2 match dataset download and building extraction. Everything after is new.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {pipeline.map((step) => (
              <div
                key={step.step}
                className="flex items-center gap-3 rounded-lg border border-white/5 bg-white/2 px-3 py-2"
              >
                <CheckCircle2 className="size-4 text-emerald-400" />
                <span className="w-6 text-xs text-slate-500">{step.step}</span>
                <span className="flex-1 text-sm text-slate-200">{step.name}</span>
                <Badge variant="secondary" className="text-[10px] uppercase">
                  {step.step <= 2 ? "yours" : "continuation"}
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardTitle>Joint Damage Scale</CardTitle>
            <CardDescription>Ordinal labels used by xBD / xView2.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {classes.map((c) => (
              <div key={c.id} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <span
                      className="size-2.5 rounded-full"
                      style={{ background: c.color }}
                    />
                    {prettyClass(c.id)}
                  </span>
                  <span className="tabular-nums text-slate-400">
                    F1 {(metrics.per_class_f1[c.id] * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress value={metrics.per_class_f1[c.id] * 100} />
                <p className="text-xs text-slate-500">{c.meaning}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <section className="space-y-3">
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <Layers3 className="size-4" />
          Tile severity (mean ordinal weight × 100)
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {events.map((event) => {
            const band = severityBand(event.severity_pred);
            return (
              <Link key={event.id} href={`/map?event=${event.id}`}>
                <Card className="h-full bg-[#0d1724] ring-white/10 transition hover:ring-orange-400/40">
                  <CardHeader className="flex-row items-start justify-between gap-3">
                    <div>
                      <CardTitle>{event.name}</CardTitle>
                      <CardDescription>
                        {event.place} · {event.date} · {event.buildings.length} buildings
                      </CardDescription>
                    </div>
                    <Badge className={band.className}>{band.label}</Badge>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="overflow-hidden rounded-lg">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={event.tile}
                        alt={`${event.name} post-disaster tile`}
                        className="h-40 w-full object-cover"
                      />
                    </div>
                    <div className="flex items-end justify-between">
                      <div>
                        <p className="text-xs text-slate-500">Predicted severity</p>
                        <p className="text-2xl font-semibold tabular-nums text-white">
                          {event.severity_pred.toFixed(1)}
                        </p>
                      </div>
                      <p className="text-xs text-slate-500">
                        labeled {event.severity_true.toFixed(1)}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      </section>

      <p className="flex items-start gap-2 text-xs text-slate-500">
        <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
        {catalog.note}
      </p>
    </div>
  );
}

function Kpi({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <Card className="bg-[#0d1724] ring-white/10">
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl tabular-nums">{value}</CardTitle>
      </CardHeader>
      <CardContent className="text-xs text-slate-500">{hint}</CardContent>
    </Card>
  );
}
