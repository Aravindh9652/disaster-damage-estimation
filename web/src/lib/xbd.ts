export type XbdMetrics = {
  source: string;
  dataset: string;
  n_total: number;
  counts: Record<string, number>;
  baseline: string;
  n_eval: number;
  metrics: {
    weighted_f1: number;
    macro_f1: number;
    per_class_f1: Record<string, number>;
    confusion_matrix: number[][];
  };
  classes: { id: string; ordinal: number; color: string }[];
  gallery: {
    uid: string;
    label: string;
    pred: string;
    crop: string;
    probs: Record<string, number>;
    correct: boolean;
  }[];
  severity_if_all_predicted: number;
  severity_if_all_labeled: number;
};

export async function loadXbdMetrics(): Promise<XbdMetrics> {
  const res = await fetch("/demo/xbd_metrics.json", { cache: "no-store" });
  if (!res.ok) throw new Error("xBD metrics not generated yet");
  return res.json();
}
