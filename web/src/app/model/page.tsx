"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useCatalog } from "@/hooks/use-catalog";
import { CLASS_ORDER, prettyClass } from "@/lib/catalog";

export default function ModelPage() {
  const { catalog, error, loading } = useCatalog();
  if (loading) return <p className="text-slate-400">Loading metrics…</p>;
  if (error || !catalog) return <p className="text-red-300">{error}</p>;

  const { metrics, history, classes } = catalog;
  const matrix = metrics.confusion_matrix;
  const maxCell = Math.max(...matrix.flat(), 1);
  const maxF1 = Math.max(...history.map((h) => h.val_weighted_f1), 0.01);
  const maxLoss = Math.max(...history.map((h) => h.val_loss), 0.01);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">ResNet50 evaluation</h1>
        <p className="mt-1 max-w-2xl text-sm text-slate-400">
          ImageNet-pretrained ResNet50, frozen backbone, ordinal-weighted cross-entropy.
          Weighted F1 is the score that matters on xBD because no-damage dominates.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardDescription>Weighted F1</CardDescription>
            <CardTitle className="text-3xl tabular-nums">
              {metrics.weighted_f1.toFixed(3)}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardDescription>Macro F1</CardDescription>
            <CardTitle className="text-3xl tabular-nums">
              {metrics.macro_f1.toFixed(3)}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardDescription>Buildings</CardDescription>
            <CardTitle className="text-3xl tabular-nums">
              {metrics.n_buildings}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardTitle>Confusion matrix</CardTitle>
            <CardDescription>Rows are true labels, columns are predictions.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
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
                                background: `rgba(249, 115, 22, ${0.08 + intensity * 0.7})`,
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
            </div>
            <p className="mt-3 text-xs text-slate-500">
              Minor vs major damage is the expected failure mode on the Joint Damage Scale —
              roofs look similar at 0.3 m, and xView2 baselines also collapse those two classes.
            </p>
          </CardContent>
        </Card>

        <Card className="bg-[#0d1724] ring-white/10">
          <CardHeader>
            <CardTitle>Training curves</CardTitle>
            <CardDescription>Frozen-backbone fine-tune, 8 epochs.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <Spark
              title="Val weighted F1"
              values={history.map((h) => h.val_weighted_f1)}
              max={maxF1}
              color="#22c55e"
            />
            <Spark
              title="Val loss"
              values={history.map((h) => h.val_loss)}
              max={maxLoss}
              color="#f97316"
            />
          </CardContent>
        </Card>
      </div>

      <Card className="bg-[#0d1724] ring-white/10">
        <CardHeader>
          <CardTitle>Per-class F1</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Class</TableHead>
                <TableHead>Ordinal</TableHead>
                <TableHead>F1</TableHead>
                <TableHead>Meaning</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {classes.map((c) => (
                <TableRow key={c.id}>
                  <TableCell className="capitalize">{prettyClass(c.id)}</TableCell>
                  <TableCell>{c.ordinal}</TableCell>
                  <TableCell className="tabular-nums">
                    {metrics.per_class_f1[c.id].toFixed(3)}
                  </TableCell>
                  <TableCell className="text-slate-400">{c.meaning}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

function Spark({
  title,
  values,
  max,
  color,
}: {
  title: string;
  values: number[];
  max: number;
  color: string;
}) {
  const w = 320;
  const h = 72;
  const pts = values
    .map((v, i) => {
      const x = (i / Math.max(values.length - 1, 1)) * w;
      const y = h - (v / max) * (h - 8) - 4;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <div>
      <p className="mb-2 text-xs text-slate-400">{title}</p>
      <svg viewBox={`0 0 ${w} ${h}`} className="h-20 w-full">
        <polyline fill="none" stroke={color} strokeWidth="2.5" points={pts} />
      </svg>
      <div className="flex justify-between text-[10px] text-slate-500">
        {values.map((v, i) => (
          <span key={i}>{v.toFixed(2)}</span>
        ))}
      </div>
    </div>
  );
}
