import json
import math
import platform
import random
import statistics
import sys
from pathlib import Path

ROOT = Path.cwd()
features = json.loads((ROOT / "data" / "idea_features.json").read_text())
spec = json.loads((ROOT / "experiment_spec.json").read_text())
rng = random.Random(spec.get("fixed_seed", 1729))
ideas = features.get("ideas", [])
selected = features.get("selected_idea", {})
trials = int(features.get("trials", 96))

def reviewer_average(idea_id):
    for review in features.get("reviewer_scores", []):
        if review.get("idea_id") == idea_id:
            return float(review.get("average_score", 5.0))
    return float(selected.get("reviewer_average", 5.0))

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def utility(idea, noise=True):
    reviewer = reviewer_average(idea.get("idea_id")) / 10.0
    evidence = float(idea.get("evidence_support", 0.4))
    novelty = 1.0 - float(idea.get("novelty_risk", 0.5))
    feasibility = float(idea.get("local_feasibility", 0.5))
    signal = 0.34 * reviewer + 0.24 * evidence + 0.24 * novelty + 0.18 * feasibility
    if noise:
        signal += rng.gauss(0.0, 0.055)
    return clamp(signal)

def rank_guided(idea, ablation=None):
    reviewer = reviewer_average(idea.get("idea_id")) / 10.0
    evidence = float(idea.get("evidence_support", 0.4))
    novelty = 1.0 - float(idea.get("novelty_risk", 0.5))
    feasibility = float(idea.get("local_feasibility", 0.5))
    weights = {
        "reviewer": 0.34 if ablation != "no_reviewer" else 0.0,
        "evidence": 0.24 if ablation != "no_evidence" else 0.0,
        "novelty": 0.24 if ablation != "no_novelty" else 0.0,
        "feasibility": 0.18,
    }
    total = sum(weights.values()) or 1.0
    return (
        weights["reviewer"] * reviewer
        + weights["evidence"] * evidence
        + weights["novelty"] * novelty
        + weights["feasibility"] * feasibility
    ) / total

if not ideas:
    ideas = [selected]

random_scores = []
guided_scores = []
ablation_scores = {"no_reviewer": [], "no_evidence": [], "no_novelty": []}

for _ in range(trials):
    pool = rng.sample(ideas, k=min(len(ideas), max(1, min(5, len(ideas)))))
    random_choice = rng.choice(pool)
    guided_choice = max(pool, key=lambda item: rank_guided(item))
    random_scores.append(utility(random_choice))
    guided_scores.append(utility(guided_choice))
    for ablation in ablation_scores:
        ablated_choice = max(pool, key=lambda item, a=ablation: rank_guided(item, a))
        ablation_scores[ablation].append(utility(ablated_choice))

random_mean = statistics.mean(random_scores)
guided_mean = statistics.mean(guided_scores)
improvement = guided_mean - random_mean
selected_utility = utility(selected, noise=False)
ablations = {
    key: {
        "mean": statistics.mean(values),
        "delta_vs_guided": statistics.mean(values) - guided_mean,
    }
    for key, values in ablation_scores.items()
}
success = improvement >= 0.025 and selected_utility >= 0.55
metrics = {
    "experiment_id": spec["experiment_id"],
    "run_id": spec["run_id"],
    "idea_id": spec["idea_id"],
    "level": spec["level"],
    "seed": spec.get("fixed_seed", 1729),
    "trials": trials,
    "random_mean": round(random_mean, 6),
    "guided_mean": round(guided_mean, 6),
    "improvement": round(improvement, 6),
    "selected_idea_utility": round(selected_utility, 6),
    "success": success,
    "ablations": {
        key: {
            "mean": round(value["mean"], 6),
            "delta_vs_guided": round(value["delta_vs_guided"], 6),
        }
        for key, value in ablations.items()
    },
    "python": sys.version.split()[0],
    "platform": platform.platform(),
}
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
(ROOT / "outputs" / "selection_scores.json").write_text(
    json.dumps(
        {
            "random_scores": random_scores[:40],
            "guided_scores": guided_scores[:40],
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(metrics, indent=2))
