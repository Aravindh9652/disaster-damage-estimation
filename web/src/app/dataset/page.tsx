"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { prettyClass } from "@/lib/catalog";
import { loadXbdMetrics, type XbdMetrics } from "@/lib/xbd";
import { useEffect, useMemo, useState } from "react";

const CLASS_ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed"] as const;
const COLORS: Record<string, string> = {
  "no-damage": "#22c55e",
  "minor-damage": "#eab308",
  "major-damage": "#f97316",
  destroyed: "#ef4444",
};

export default function DatasetPage() {
  const [data, setData] = useState<XbdMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    loadXbdMetrics()
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  const gallery = useMemo(() => {
    if (!data) return [];
    if (filter === "all") return data.gallery;
    if (filter === "mistakes") return data.gallery.filter((g) => !g.correct);
    return data.gallery.filter((g) => g.label === filter);
  }, [data, filter]);

  if (!data && !error) return <p className="text-slate-400">Loading your Drive crops…</p>;
  if (error || !data) {
    return (
      <div className="space-y-3">
        <h1 className="text-2xl font-semibold text-white">Your xBD crops</h1>
        <p className="text-sm text-amber-200">
          {error}. Download the Drive folder, then run{" "}
          <code className="rounded bg-white/10 px-1">python ml/baseline_eval.py</code>.
        </p>
      </div>
    );
  }

  const maxCount = Math.max(...Object.values(data.counts), 1);
  const matrix = data.metrics.confusion_matrix;
  const maxCell = Math.max(...matrix.flat(), 1);

  return (
    <div className="space-y-6">
      <div>
        <Badge variant="secondary" className="bg-emerald-500/15 text-emerald-200">
          Loaded from your Google Drive folder
        </Badge>
        <h1 className="mt-2 text-2xl font-semibold text-white">xBD building crops</h1>
        <p className="mt-1 max-w-2xl text-sm text-slate-400">
          {data.dataset}. {data.n_total.toLocaleString()} chips already extracted
          into four Joint Damage Scale folders. This page is the continuation on
          <em> your</em> data, not the synthetic map tiles.
        </p>
        <a
          href={data.source}
          className="mt-2 inline-block text-xs text-orange-300 hover:underline"
          target="_blank"
          rel="noreferrer"
        >
          Open Drive folder
        </a>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {CLASS_ORDER.map((id) => (
          <Card key={id} className="bg-[#0d1724] ring-white/10">
            <CardHeader className="pb-2">
              <CardDescription className="capitalize">{prettyClass(id)}</CardDescription>
              <CardTitle className="text-2xl tabular-nums">
                {(data.counts[id] ?? 0).toLocaleString()}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full"
                  style={{
                    width: `${((data.counts[id] ?? 0) / maxCount) * 100}%`,
                    background: COLORS[id],
                  }}
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardTitle>CPU baseline on your chips</CardTitle>
            <CardDescription>
              {data.baseline}. Weighted F1 {data.metrics.weighted_f1.toFixed(3)} on{" "}
              {data.n_eval} held-out buildings. Run ResNet50 with{" "}
              <code>python ml/train.py --root ..</code> after splits.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="mb-4 grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-slate-500">Weighted F1</p>
                <p className="text-2xl tabular-nums">{data.metrics.weighted_f1.toFixed(3)}</p>
              </div>
              <div>
                <p className="text-slate-500">Macro F1</p>
                <p className="text-2xl tabular-nums">{data.metrics.macro_f1.toFixed(3)}</p>
              </div>
            </div>
            <table className="w-full text-center text-xs">
              <thead>
                <tr>
                  <th />
                  {CLASS_ORDER.map((c) => (
                    <th key={c} className="px-1 pb-2 font-normal text-slate-500">
                      {prettyClass(c)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {CLASS_ORDER.map((trueClass, i) => (
                  <tr key={trueClass}>
                    <th className="pr-2 text-right font-normal text-slate-500">
                      {prettyClass(trueClass)}
                    </th>
                    {matrix[i].map((n, j) => {
                      const intensity = n / maxCell;
                      return (
                        <td key={`${i}-${j}`} className="p-1">
                          <div
                            className="flex h-12 items-center justify-center rounded tabular-nums"
                            style={{
                              background: `rgba(34, 197, 94, ${0.08 + intensity * 0.7})`,
                              color: intensity > 0.45 ? "#fff" : "#cbd5e1",
                            }}
                          >
                            {n}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <ul className="mt-3 space-y-1 text-xs text-slate-400">
              {CLASS_ORDER.map((c) => (
                <li key={c} className="flex justify-between">
                  <span>{prettyClass(c)}</span>
                  <span className="tabular-nums">
                    F1 {(data.metrics.per_class_f1[c] * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardTitle>Severity from your labels</CardTitle>
            <CardDescription>
              Mean Joint Damage Scale weight × 100 on the evaluation split.
              Your Drive subset is damage-heavy (no-damage is the smallest class),
              so the index sits higher than a typical full xBD holdout.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Labeled severity</span>
              <span className="text-2xl tabular-nums text-white">
                {data.severity_if_all_labeled.toFixed(1)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Baseline predicted</span>
              <span className="text-2xl tabular-nums">
                {data.severity_if_all_predicted.toFixed(1)}
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Next step on GPU: freeze ResNet50, train the head on{" "}
              <code>data/xbd/crops</code>, then replace this baseline.
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-wrap gap-2">
        {["all", "mistakes", ...CLASS_ORDER].map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setFilter(id)}
            className={`rounded-md px-3 py-1 text-xs ${
              filter === id ? "bg-white/15 text-white" : "bg-white/5 text-slate-400"
            }`}
          >
            {id === "all" || id === "mistakes" ? id : prettyClass(id)}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        {gallery.map((g) => (
          <figure
            key={g.uid + g.crop}
            className="overflow-hidden rounded-lg ring-1 ring-white/10"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={g.crop} alt={g.uid} className="aspect-square w-full object-cover" />
            <figcaption className="space-y-0.5 px-2 py-1.5 text-[10px]">
              <p className="truncate text-slate-300">{g.uid.slice(0, 8)}</p>
              <p>
                <span style={{ color: COLORS[g.pred] }}>{prettyClass(g.pred)}</span>
                {!g.correct && (
                  <span className="text-slate-500">
                    {" "}
                    · true {prettyClass(g.label)}
                  </span>
                )}
              </p>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
