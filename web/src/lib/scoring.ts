export type PriorityLevel = "critical" | "high" | "moderate" | "low";

export type PriorityConfiguration = {
  weights: { severity: number; damage_area: number; impact: number };
  cutoffs: { critical: number; high: number; moderate: number };
};

export const DEFAULT_PRIORITY_CONFIGURATION: PriorityConfiguration = {
  weights: { severity: 0.5, damage_area: 0.3, impact: 0.2 },
  cutoffs: { critical: 60, high: 35, moderate: 15 },
};

export function priorityLevel(
  score: number,
  cutoffs: PriorityConfiguration["cutoffs"] = DEFAULT_PRIORITY_CONFIGURATION.cutoffs,
): PriorityLevel {
  if (score >= cutoffs.critical) return "critical";
  if (score >= cutoffs.high) return "high";
  if (score >= cutoffs.moderate) return "moderate";
  return "low";
}

export function priorityScore(
  severity: number,
  damageArea: number,
  impactFactor = 1,
  configuration: PriorityConfiguration = DEFAULT_PRIORITY_CONFIGURATION,
) {
  const severityFrac = severity > 1 ? severity / 100 : severity;
  const area = Math.min(1, Math.max(0, damageArea));
  const normalizedImpactFactor = Math.min(1, Math.max(0, impactFactor));
  const impact = severityFrac * normalizedImpactFactor;
  const score = 100 * (
    configuration.weights.severity * severityFrac
    + configuration.weights.damage_area * area
    + configuration.weights.impact * impact
  );
  const rounded = Math.round(score * 100) / 100;
  const level = priorityLevel(rounded, configuration.cutoffs);
  return {
    priority_score: rounded,
    priority_level: level,
    components: { severity: severityFrac, damage_area: area, impact, impact_factor: normalizedImpactFactor },
    formula:
      "100 * (w_severity * normalized_severity + w_damage_area * normalized_damage_area + w_impact * normalized_severity * normalized_impact_factor)",
    priority_rationale: `${level} project-defined decision-support score (${rounded.toFixed(2)}/100; cutoffs ${JSON.stringify(configuration.cutoffs)}).`,
  };
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
