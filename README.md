# makefun-gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sheeran154/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sheeran154/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), after the first hour. One program,
`main.ts` (Node 22, standard library only, no network, no dependencies):
it reads a case file, learns the launch-day shape from past big launches,
and writes five quantiles (5 / 25 / 50 / 75 / 95%) for hours 2–24.

## How it works

A launch day's hourly curve is mostly determined by how big the hour-1
print already is. Across past big launches, `log(peak / hour-1)` falls
almost linearly with `log(hour-1)` (correlation ≈ −0.94): games that open
huge at 00:00Z — a simultaneous global unlock — stay flat, while games
that open small — rolling midnight, only New Zealand / Australia / Asia
awake at 00:00Z — ramp several-fold into the US unlocks (05:00Z Eastern,
08:00Z Pacific) and the evening peak.

So for each forecast hour the program fits, over the most comparable past
launches,

```
log(viewers[h] / viewers[1]) = a_h + b_h * log(viewers[1])
```

by ordinary least squares, predicts at this case's hour-1 print, and takes
the residual spread as the lognormal quantile width (smoothed across
neighbouring hours, since each hour's spread comes from only ~a dozen
launches). Training uses the rolling-midnight, tier-"comparable" launches
only — early-access, simultaneous-global and weak analogues open with a
different shape — with fallbacks if that subset is too small. The target
game's own pre-release category history is never training data (its
premiere spike would wreck the fit), and the hour-1 print is never
extrapolated above the biggest launch actually seen.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` file in the input directory (else `/input/`) and writes
one `/output/<id>.out` per case: exactly `{"hours": [...23 rows...]}`, each
row `{"hour": h, "q": [q05, q25, q50, q75, q95]}` for hours 2–24.
Deterministic — two runs produce byte-identical output — and fast
(~0.2 s for all four practice cases).

## Practice results

Practice answers were pulled from the same source the kit cites
(SullyGnome hourly Twitch category averages) and scored with the
competition's pinball loss on the log scale. Mean loss over the four
practice launches (Wolverine, Red Dead Redemption 2, God of War
Ragnarök, Marvel's Spider-Man 2):

| case | this entry | house baseline |
| ---- | ---------: | -------------: |
| p00 Wolverine | 0.187 | 0.744 |
| p01 RDR2 | 0.122 | 0.287 |
| p02 God of War Ragnarök | 0.134 | 0.215 |
| p03 Spider-Man 2 | 0.223 | 0.303 |
| **mean** | **0.166** | **0.387** |

Beats the house baseline on every practice case, at less than half its
mean loss. The prize needs a loss strictly below the best house entry on
the final (GTA VI, 2026-11-19).

---

<sub>Entered on [make.fun](https://make.fun/?ref=nvhyrpj1&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
