export type DamageClass =
  | "no-damage"
  | "minor-damage"
  | "major-damage"
  | "destroyed";

export type Building = {
  uid: string;
  disaster_id: string;
  disaster: string;
  tile: string;
  path: string;
  crop: string;
  label: DamageClass;
  pred: DamageClass;
  label_idx: number;
  probs: Record<DamageClass, number>;
  polygon: { x: number; y: number }[];
  centroid: { x: number; y: number };
};

export type DisasterEvent = {
  id: string;
  name: string;
  place: string;
  type: string;
  date: string;
  tile: string;
  width: number;
  height: number;
  buildings: Building[];
  counts_true: Record<DamageClass, number>;
  counts_pred: Record<DamageClass, number>;
  severity_true: number;
  severity_pred: number;
};

export type Catalog = {
  project: string;
  continuation: string;
  note: string;
  classes: {
    id: DamageClass;
    ordinal: number;
    color: string;
    meaning: string;
  }[];
  pipeline: { step: number; name: string; status: string }[];
  metrics: {
    weighted_f1: number;
    macro_f1: number;
    per_class_f1: Record<DamageClass, number>;
    confusion_matrix: number[][];
    n_buildings: number;
    n_events: number;
  };
  history: {
    epoch: number;
    train_loss: number;
    val_loss: number;
    train_weighted_f1: number;
    val_weighted_f1: number;
  }[];
  events: DisasterEvent[];
};

export const CLASS_ORDER: DamageClass[] = [
  "no-damage",
  "minor-damage",
  "major-damage",
  "destroyed",
];

export function prettyClass(id: string) {
  return id.replaceAll("-", " ");
}

export function severityBand(score: number) {
  if (score >= 60) return { label: "Critical", className: "bg-red-500/15 text-red-300" };
  if (score >= 35) return { label: "Severe", className: "bg-orange-500/15 text-orange-300" };
  if (score >= 15) return { label: "Moderate", className: "bg-yellow-500/15 text-yellow-300" };
  return { label: "Low", className: "bg-emerald-500/15 text-emerald-300" };
}

export async function loadCatalog(): Promise<Catalog> {
  const res = await fetch("/demo/catalog.json", { cache: "no-store" });
  if (!res.ok) throw new Error("Could not load damage catalog");
  return res.json();
}
