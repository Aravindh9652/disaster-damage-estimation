export type PriorityLevel = "critical" | "high" | "moderate" | "low";

export function priorityLevel(score: number): PriorityLevel {
  if (score >= 60) return "critical";
  if (score >= 35) return "high";
  if (score >= 15) return "moderate";
  return "low";
}

export function priorityScore(severity: number, damageArea: number, confidence: number) {
  const severityFrac = severity > 1 ? severity / 100 : severity;
  const area = Math.min(1, Math.max(0, damageArea));
  const conf = Math.min(1, Math.max(0, confidence));
  const impact = severityFrac * conf;
  const score = 100 * (0.5 * severityFrac + 0.3 * area + 0.2 * impact);
  const rounded = Math.round(score * 100) / 100;
  return { priority_score: rounded, priority_level: priorityLevel(rounded) };
}

const AREA_WEIGHTS: Record<string, number> = {
  "no-damage": 0.02,
  "minor-damage": 0.25,
  "major-damage": 0.6,
  destroyed: 0.95,
};

export function zoneDamageArea(labels: string[]) {
  if (labels.length === 0) return 0;
  const total = labels.reduce((sum, label) => sum + (AREA_WEIGHTS[label] ?? 0), 0);
  return total / labels.length;
}

export function levelClass(level: PriorityLevel) {
  if (level === "critical") return "bg-red-500/15 text-red-300";
  if (level === "high") return "bg-orange-500/15 text-orange-300";
  if (level === "moderate") return "bg-yellow-500/15 text-yellow-300";
  return "bg-emerald-500/15 text-emerald-300";
}
