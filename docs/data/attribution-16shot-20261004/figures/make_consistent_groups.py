"""Show LR groups ranked in the top ten of sixteen in every held-out fold.

The selection rule is fixed across cohorts. Export every candidate group,
including excluded groups, so selection can be audited from aggregate data.
"""

import csv
import statistics

from make_figures import COHORTS, OUTPUT, display_label, plt, read_results
from matplotlib.colors import Normalize


DESTINATION = OUTPUT / "consistent_groups"
TOP_K = 10


def group_records(rankings, folds):
    records = []
    for cohort in COHORTS:
        groups = rankings[(cohort, "group")]
        assert len(groups) == 16
        for row in groups:
            values = [folds[(cohort, "group", row["feature_name"], fold)]
                      for fold in range(5)]
            ranks = [int(value["rank_in_fold"]) if value["rank_in_fold"] else None
                     for value in values]
            scores = [float(value["mean_absolute_pp"]) for value in values]
            mean = float(row["mean_absolute_pp"])
            sd = float(row["fold_sd_absolute_pp"])
            assert abs(mean - statistics.mean(scores)) < 1e-10
            assert abs(sd - statistics.stdev(scores)) < 1e-10
            count = sum(rank is not None and rank <= TOP_K for rank in ranks)
            assert abs(count / 5 - float(row["top10_fold_fraction"])) < 1e-10
            record = {
                "cohort": cohort, "variant": "PathoTME-LR", "class_index": 1,
                "feature_name": row["feature_name"],
                "display_label": display_label(row["feature_name"], "group"),
                "mean_absolute_pp": mean, "fold_sd_absolute_pp": sd,
                **{f"fold_{fold}_rank": rank for fold, rank in enumerate(ranks)},
                **{f"fold_{fold}_absolute_pp": score for fold, score in enumerate(scores)},
                "top10_folds": count,
                "top5_folds": sum(rank is not None and rank <= 5 for rank in ranks),
                "selected_top10_all_five_folds": count == 5,
            }
            records.append(record)
    return sorted(records, key=lambda row: (list(COHORTS).index(row["cohort"]),
                                           -row["mean_absolute_pp"], row["feature_name"]))


def render(records):
    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(2, 2, left=0.06, right=0.95, top=0.82,
                           bottom=0.17, wspace=0.24, hspace=0.20)
    colors = plt.get_cmap("Blues")
    norm = Normalize(vmin=1, vmax=16)
    max_mean = max((row["mean_absolute_pp"] for row in records
                    if row["selected_top10_all_five_folds"]), default=1)
    for cell, (cohort, (title, positive_class)) in zip(grid, COHORTS.items()):
        axis = fig.add_subplot(cell)
        selected = [row for row in records
                    if row["cohort"] == cohort and row["selected_top10_all_five_folds"]]
        axis.set_axis_off()
        axis.text(0, 1.08, title, fontsize=14, weight="bold", color="#172B41",
                  transform=axis.transAxes)
        axis.text(0, 1.01, f"Predicted {positive_class} probability · {len(selected)} of 16 groups",
                  fontsize=9, color="#536578", transform=axis.transAxes)
        if not selected:
            axis.text(0, 0.78, "No group meets the selection rule", transform=axis.transAxes)
            continue
        axis.text(0, 0.90, "GROUP · MEAN (pp)", fontsize=8, weight="bold", color="#52687B",
                  transform=axis.transAxes)
        axis.text(0.51, 0.90, "FOLD RANK", fontsize=8, weight="bold", color="#52687B",
                  transform=axis.transAxes)
        axis.text(0.92, 0.90, "TOP 5", fontsize=8, weight="bold", color="#52687B",
                  transform=axis.transAxes, ha="center")
        for fold in range(5):
            axis.text(0.53 + fold * 0.075, 0.82, str(fold), ha="center",
                      fontsize=8, color="#536578", transform=axis.transAxes)
        for index, row in enumerate(selected):
            y = 0.70 - index * 0.15
            axis.text(0, y, row["display_label"], fontsize=9,
                      color="#26374A", va="center", transform=axis.transAxes)
            mean = row["mean_absolute_pp"]
            axis.add_patch(plt.Rectangle((0, y - 0.066), 0.35 * mean / max_mean,
                                         0.028, facecolor="#147A7E",
                                         transform=axis.transAxes))
            axis.text(0.44, y - 0.052, f"{mean:.2f}", ha="right", va="center",
                      fontsize=8, color="#26374A", transform=axis.transAxes)
            for fold in range(5):
                rank = row[f"fold_{fold}_rank"]
                shade = colors(0.12 + 0.78 * (1 - norm(rank)))
                axis.add_patch(plt.Rectangle((0.497 + fold * 0.075, y - 0.039),
                                             0.066, 0.078, facecolor=shade,
                                             edgecolor="white", transform=axis.transAxes))
                axis.text(0.53 + fold * 0.075, y, str(rank), ha="center", va="center",
                          fontsize=9, color="white" if rank <= 6 else "#26374A",
                          transform=axis.transAxes)
            axis.text(0.92, y, f"{row['top5_folds']}/5", fontsize=9,
                      color="#26374A", va="center", ha="center", transform=axis.transAxes)
        axis.text(0, 0.045, f"{16 - len(selected)} groups do not meet this rule",
                  fontsize=8, color="#536578", transform=axis.transAxes)
    fig.text(0.06, 0.94, "TME groups ranked in the top 10 in every fold",
             fontsize=20, weight="bold", color="#172B41")
    fig.text(0.06, 0.90, "PathoTME-LR · 16-shot · five held-out folds · rank 1 = highest attribution",
             fontsize=11, color="#536578")
    fig.text(0.06, 0.09, "Selection: rank ≤10 of 16 groups in all five folds. Darker cells indicate higher ranks.",
             fontsize=9, color="#536578")
    fig.text(0.06, 0.065, "Rank consistency does not establish stable effect sizes or predictive usefulness. All fold scores are retained in the CSV.",
             fontsize=8.5, color="#536578")
    fig.text(0.06, 0.040, "Mean attribution: absolute patient-level probability change after joint group replacement, averaged over five folds (pp).",
             fontsize=8.5, color="#536578")
    for extension in ("png", "svg", "pdf"):
        path = DESTINATION / f"lr_consistent_groups_16shot.{extension}"
        fig.savefig(path, dpi=300, facecolor="white",
                    metadata={"Title": "PathoTME-LR 16-shot groups ranked in the top ten in every fold"})
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.bbox": "standard"})
    rankings, folds = read_results()
    records = group_records(rankings, folds)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with (DESTINATION / "all_lr_group_consistency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    render(records)
    for cohort in COHORTS:
        selected = [row for row in records if row["cohort"] == cohort
                    and row["selected_top10_all_five_folds"]]
        print(f"{cohort}: {len(selected)} selected groups", flush=True)


if __name__ == "__main__":
    main()
