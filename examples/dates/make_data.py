"""Generate the date-normalization benchmark (deterministic for a given seed).

Task: find the date in a short text and return it as YYYY-MM-DD. The data follows conventions
that the weak baseline prompt does not state, so the optimizer has to discover them from
failures, one round at a time:

  * the answer is the bare ISO date, nothing else;
  * numeric dates are day-first (European): 04/03/2025 is 4 March, not April 3;
  * two-digit years are 20YY;
  * the date is often embedded in a sentence with other numbers (times, amounts).

Each example's metadata records its format family, so splits can be stratified by it.

    python examples/dates/make_data.py            # writes data.jsonl next to this file
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def render(d: dt.date, family: str, rng: random.Random) -> str:
    m, day, y = d.month, d.day, d.year
    if family == "written_us":
        return f"{MONTHS[m - 1]} {day}, {y}"
    if family == "written_eu":
        return rng.choice([f"{day} {MONTHS[m - 1]} {y}", f"{ordinal(day)} of {MONTHS[m - 1]}, {y}",
                           f"{day} {MONTHS[m - 1][:3]} {y}"])
    if family == "numeric_ambiguous":  # day and month both <= 12: only the convention decides
        sep = rng.choice(["/", "-", "."])
        return f"{day:02d}{sep}{m:02d}{sep}{y}"
    if family == "numeric_unambiguous":  # day > 12
        sep = rng.choice(["/", "."])
        return f"{day}{sep}{m}{sep}{y}"
    if family == "two_digit_year":
        return f"{day:02d}/{m:02d}/{y % 100:02d}"
    if family == "year_first":
        return f"{y}/{m:02d}/{day:02d}"
    raise ValueError(family)


SENTENCES = [
    "Your appointment is confirmed for {d} at {h}:{mi:02d}.",
    "Invoice #{n} of {d}: total due {amt} EUR.",
    "The package left the warehouse on {d}, tracking code {n}.",
    "Meeting moved to {d} (room {r}).",
    "Payment of {amt} EUR received on {d}.",
    "{d}",
    "Signed in Milan on {d}.",
    "Flight AZ{n} departs {d} from gate {r}.",
]


def make(seed: int = 0, per_family: int = 20) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    families = ["written_us", "written_eu", "numeric_ambiguous", "numeric_unambiguous",
                "two_digit_year", "year_first"]
    for family in families:
        for i in range(per_family):
            while True:
                d = dt.date(rng.randint(2019, 2029), rng.randint(1, 12), rng.randint(1, 28))
                if family == "numeric_ambiguous" and (d.day > 12 or d.day == d.month):
                    continue
                if family == "numeric_unambiguous" and d.day <= 12:
                    continue
                if family == "two_digit_year" and (d.day <= 12 and d.day != d.month):
                    # keep two-digit-year examples about the year, not the day/month order
                    continue
                break
            text = rng.choice(SENTENCES).format(
                d=render(d, family, rng), h=rng.randint(8, 18), mi=rng.choice([0, 15, 30, 45]),
                n=rng.randint(100, 9999), amt=f"{rng.randint(10, 999)}.{rng.randint(0, 99):02d}",
                r=rng.randint(1, 40),
            )
            rows.append({"id": f"{family}-{i:02d}", "inputs": {"text": text},
                         "expected": d.isoformat(), "metadata": {"family": family}})
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-family", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).with_name("data.jsonl")))
    args = ap.parse_args()
    rows = make(args.seed, args.per_family)
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
