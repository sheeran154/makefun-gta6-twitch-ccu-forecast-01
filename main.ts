// gta6-twitch-ccu-forecast-01 — launch-day Twitch CCU forecaster.
//
// Idea: a launch day's hourly curve is mostly determined by how big the
// hour-1 print already is. Across past big launches, log(peak / hour-1)
// falls almost linearly with log(hour-1) (corr ≈ -0.94): games that open
// huge at 00:00Z (simultaneous global unlock) stay flat, while games that
// open small (rolling midnight — only NZ/AU/Asia awake at 00:00Z) ramp
// several-fold into the US unlocks and the evening peak.
//
// So for each forecast hour we fit, over the most comparable past launches,
//   log(viewers[h] / viewers[1]) = a_h + b_h * log(viewers[1])
// by ordinary least squares, predict at this case's hour-1 print, and take
// the residual spread as the lognormal quantile width. The training set is
// the rolling-midnight, tier-"comparable" launches (early-access,
// simultaneous-global and weak analogues are excluded — they open with a
// different shape), with fallbacks if that subset is too small.
//
// Deterministic: no randomness, no clock, no network. Stdlib only.

import * as fs from "node:fs";
import * as path from "node:path";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
// Standard-normal quantiles for the five levels.
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

const MIN_SIGMA = 0.08; // never claim more precision than this (log units)
const MIN_TRAIN = 5; // smallest training set we trust for the regression

interface HourPoint {
  from: string;
  viewers: number;
  channels?: number;
}

interface Comparable {
  name?: string;
  day_start?: string;
  unlock?: string;
  hourly?: HourPoint[];
}

interface CaseFile {
  id?: string;
  game?: { name?: string };
  day_start?: string;
  known?: Array<{ hour?: number; viewers?: number; channels?: number }>;
  history_hourly?: HourPoint[];
  comparables?: Comparable[];
}

interface TrainRow {
  h1: number;
  day: number[]; // viewers at hours 0..23 relative to day_start
  tier: string;
  rolling: boolean;
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

// Tier parsed out of a comparable's free-text `unlock` field.
function tierOf(unlock: string): string {
  if (unlock.includes("Tier: comparable")) return "comparable";
  if (unlock.includes("early-access-contaminated")) return "early-access";
  if (unlock.includes("Tier: weak")) return "weak";
  return "other";
}

function isRolling(unlock: string): boolean {
  return /midnight local|rolling/i.test(unlock);
}

// Extract one comparable's release-day hourly curve (hours 0..23), or null
// when the curve is incomplete — SullyGnome skips unsampled hours, and the
// target game's own pre-release history is sparse by design.
function dayCurve(comp: Comparable): number[] | null {
  if (!comp.day_start || !Array.isArray(comp.hourly)) return null;
  const start = Date.parse(comp.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour = new Map<number, number>();
  for (const p of comp.hourly) {
    if (!p || !isFiniteNumber(p.viewers) || p.viewers <= 0) continue;
    const t = Date.parse(p.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.round((t - start) / 3600000);
    if (h >= 0 && h < 24) byHour.set(h, p.viewers);
  }
  const day: number[] = [];
  for (let h = 0; h < 24; h++) {
    const v = byHour.get(h);
    if (v === undefined || v <= 0) return null;
    day.push(v);
  }
  return day;
}

// The target game's own pre-release category history is not a launch and
// must never enter training (its premiere spike would wreck the fit).
function isTargetHistory(comp: Comparable, gameName: string): boolean {
  const unlock = comp.unlock ?? "";
  if (unlock.includes("target-history")) return true;
  const name = (comp.name ?? "").toLowerCase();
  const game = gameName.toLowerCase();
  return game.length > 3 && name.includes(game);
}

function collectTraining(c: CaseFile): TrainRow[] {
  const gameName = c.game?.name ?? "";
  const rows: TrainRow[] = [];
  for (const comp of c.comparables ?? []) {
    if (isTargetHistory(comp, gameName)) continue;
    const day = dayCurve(comp);
    if (!day) continue;
    const h1 = day[0];
    if (!isFiniteNumber(h1) || h1 <= 0) continue;
    const unlock = comp.unlock ?? "";
    rows.push({ h1, day, tier: tierOf(unlock), rolling: isRolling(unlock) });
  }
  return rows;
}

function pickSubset(rows: TrainRow[]): TrainRow[] {
  const rollingComparable = rows.filter((r) => r.tier === "comparable" && r.rolling);
  if (rollingComparable.length >= MIN_TRAIN) return rollingComparable;
  const comparable = rows.filter((r) => r.tier === "comparable");
  if (comparable.length >= MIN_TRAIN) return comparable;
  return rows;
}

function ols(xs: number[], ys: number[]): { a: number; b: number } {
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return { a: my - b * mx, b };
}

// Hour-1 print: the `known` entry for hour 1, else the last known value.
function hourOnePrint(c: CaseFile): number {
  const known = c.known ?? [];
  for (const k of known) {
    if (k.hour === 1 && isFiniteNumber(k.viewers) && k.viewers > 0) return k.viewers;
  }
  for (const k of known) {
    if (isFiniteNumber(k.viewers) && k.viewers > 0) return k.viewers;
  }
  const hist = c.history_hourly ?? [];
  for (let i = hist.length - 1; i >= 0; i--) {
    if (isFiniteNumber(hist[i].viewers) && hist[i].viewers > 0) return hist[i].viewers;
  }
  return 1;
}

// Flat house-style fallback when there is nothing to learn from.
function flatForecast(h1: number): Array<{ hour: number; q: number[] }> {
  const hours: Array<{ hour: number; q: number[] }> = [];
  for (let h = 2; h <= 24; h++) {
    const s = 0.15 * Math.sqrt(h - 1);
    const q = LEVELS.map((_, j) => Math.max(1, Math.round(h1 * Math.exp(Z[j] * s))));
    hours.push({ hour: h, q: enforceOrder(q) });
  }
  return hours;
}

function enforceOrder(q: number[]): number[] {
  const out = [...q];
  for (let i = 1; i < out.length; i++) {
    if (out[i] < out[i - 1]) out[i] = out[i - 1];
  }
  return out;
}

function forecastCase(c: CaseFile): Array<{ hour: number; q: number[] }> {
  const h1 = hourOnePrint(c);
  const train = pickSubset(collectTraining(c));
  if (train.length < 2) return flatForecast(h1);

  const logH1 = train.map((r) => Math.log(r.h1));
  // Never extrapolate above the biggest launch we have actually seen:
  // beyond it the regression would forecast a day that falls all day,
  // which no mega-launch does. (The low end is left alone — small openers
  // genuinely do ramp the hardest.)
  const maxLogH1 = Math.max(...logH1);
  const at = Math.min(Math.log(h1), maxLogH1);

  const hours: Array<{ hour: number; q: number[] }> = [];
  // First pass: per-hour median forecast and residual spread.
  const preds: number[] = [];
  const sigmas: number[] = [];
  for (let h = 1; h < 24; h++) {
    const ys = train.map((r) => Math.log(r.day[h] / r.h1));
    const { a, b } = ols(logH1, ys);
    preds.push(a + b * at);
    let ss = 0;
    let mean = 0;
    for (let i = 0; i < ys.length; i++) mean += ys[i] - (a + b * logH1[i]);
    mean /= ys.length;
    for (let i = 0; i < ys.length; i++) {
      const d = ys[i] - (a + b * logH1[i]) - mean;
      ss += d * d;
    }
    const sd = Math.sqrt(ss / Math.max(1, ys.length - 2));
    sigmas.push(Math.max(sd, MIN_SIGMA));
  }
  // Second pass: the true spread varies smoothly through the day, but each
  // hour's estimate comes from only ~a dozen launches, so average each
  // hour's spread with its neighbours before using it. (A single noisy
  // hour must not get an absurdly narrow — or wide — interval.)
  const smooth: number[] = sigmas.map((_, i) => {
    const lo = Math.max(0, i - 1);
    const hi = Math.min(sigmas.length - 1, i + 1);
    let sum = 0;
    for (let k = lo; k <= hi; k++) sum += sigmas[k];
    return sum / (hi - lo + 1);
  });
  for (let h = 1; h < 24; h++) {
    const pred = preds[h - 1];
    const sigma = smooth[h - 1];
    const q = LEVELS.map((_, j) => {
      const v = h1 * Math.exp(pred + Z[j] * sigma);
      if (!Number.isFinite(v)) return 1;
      return Math.min(1e9, Math.max(1, Math.round(v)));
    });
    hours.push({ hour: h + 1, q: enforceOrder(q) });
  }
  return hours;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output/";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const id = file.slice(0, -3);
    const c = JSON.parse(fs.readFileSync(path.join(inputDir, file), "utf8")) as CaseFile;
    const hours = forecastCase(c);
    const out = { hours };
    fs.writeFileSync(path.join(outputDir, `${id}.out`), JSON.stringify(out), "utf8");
  }
}

main();
