"""Generated LaTeX tables for the IEEE paper (paper/ieee/sections/gen_*.tex), from the same sources as the pandoc draft:
runs/analysis/datasets/report.md (per-class audit), research_log.md section 27 (run registry), the final-phase run folders
(val best from metrics.csv) and runs/analysis/{g29,paper} (test scores).
  .venv/bin/python tools/build_ieee_tables.py
"""

import csv
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "ieee", "sections")
RUNS = os.path.join(ROOT, "runs", "detect", "BladeYOLO_WindSurface")
ARCH = os.path.join(ROOT, "runs", "analysis", "archive", "kaggle_a0s1")


def tex(s):
    s = s.strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`([^`]+)`", lambda m: "\\texttt{" + m.group(1).replace("_", "\\_\\allowbreak ") + "}", s)
    parts = re.split(r"(\\texttt\{[^}]*\}|\\textbf\{)", s)
    out = []
    for p in parts:
        if p.startswith("\\texttt") or p == "\\textbf{":
            out.append(p)
        else:
            out.append(p.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
                       .replace("—", "--").replace("–", "--").replace("−", "$-$").replace("×", "$\\times$")
                       .replace("≥", "$\\geq$").replace("≤", "$\\leq$").replace("→", "$\\rightarrow$")
                       .replace("~", "$\\sim$").replace("“", "``").replace("”", "''").replace("§", "\\S"))
    return "".join(out)


def audit():
    names = {"wtbd": "WTBD", "ws": "Wind Surface", "wt": "WTBlade", "bj": "Beijing", "wtbs": "WTBs2025"}
    rows = []
    for line in open(os.path.join(ROOT, "runs", "analysis", "datasets", "report.md"), encoding="utf-8"):
        p = [s.strip() for s in line.split("|")]
        if len(p) > 11 and p[1] in names and p[2] != "class":
            rows.append(" & ".join([names[p[1]], tex(p[2]), p[3], p[4], p[6], p[7], tex(p[8]), tex(p[10]), p[11]]) + " \\\\")
    body = "\n".join(rows)
    return ("\\begin{longtable}{@{}l l r r r r r r r@{}}\n"
            "\\caption{Content audit by native class (\\texttt{runs/analysis/datasets/report.md}). \\emph{Learnable} boxes are those kept "
            "after deduplication. \\emph{Purity} is mutual crop retrieval within the class. Median box size is in pixels at 640. The check "
            "export of Beijing is omitted because it is the same data.}\\label{tab:perclass-audit}\\\\\n\\toprule\n"
            "Dataset & Class & Raw img & Raw boxes & Unique & Scenes & Learnable & Median px & Purity \\\\\n\\midrule\n\\endfirsthead\n"
            "\\toprule Dataset & Class & Raw img & Raw boxes & Unique & Scenes & Learnable & Median px & Purity \\\\\\midrule\n\\endhead\n"
            + body + "\n\\bottomrule\n\\end{longtable}\n")


def registry():
    lines = open(os.path.join(ROOT, "research_log.md"), encoding="utf-8").read().split("\n")
    i = lines.index("## 27. Run registry")
    while not lines[i].startswith("| Run folder"):
        i += 1
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        p = [tex(x) for x in l.strip().strip("|").split("|")]
        rows.append(" & ".join(p) + " \\\\")
    return ("{\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{@{}p{0.2\\textwidth} p{0.12\\textwidth} p{0.15\\textwidth} p{0.06\\textwidth} p{0.1\\textwidth} p{0.13\\textwidth} p{0.12\\textwidth}@{}}\n"
            "\\caption{Registry of the runs before the final phase, as kept in the research log (section 27 there; section references "
            "inside this table are to the log). Scores are fractions, not percent. WS: Wind Surface.}\\label{tab:registry}\\\\\n\\toprule\n"
            "Run folder & Data & Model & Epochs & Val best (50 / 50:95) & Test (50 / 50:95) & Notes \\\\\n\\midrule\n\\endfirsthead\n"
            "\\toprule Run folder & Data & Model & Epochs & Val best & Test & Notes \\\\\\midrule\n\\endhead\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n}\n")


def val_best(path):
    rows = [r for r in csv.DictReader(open(os.path.join(path, "metrics.csv"))) if r.get("val/ema_mAP_50_95")]
    r = max(rows, key=lambda r: float(r["val/ema_mAP_50_95"]))
    n = max(int(x["epoch"]) for x in rows) + 1
    return n, int(r["epoch"]) + 1, 100 * float(r["val/ema_mAP_50"]), 100 * float(r["val/ema_mAP_50_95"])


def final_runs():
    off = json.load(open(os.path.join(ROOT, "runs", "analysis", "paper", "official.json")))

    def o(ds, run):
        return "{:.1f} / {:.1f}".format(*[100 * v for v in off[ds]["runs"][run]["all"]["coco"]])

    # test scores: tables of the paper (G27/G28 from the pandoc draft's registry; G29 from tools/g29_score.py and the
    # per-seed rescoring with the same evaluator)
    spec = [
        ("rfdetrm768cos_wtbd_group_full", RUNS, "WTBD-group, A0, s0", "55.3 / 33.2"),
        ("rfdetrm768cos_wtbd_group_full_s1", ARCH, "WTBD-group, A0, s1 (T4)", "53.2 / 31.2"),
        ("rfdetrm768cos_wtbd_group_full_s1", RUNS, "WTBD-group, A0, s1 (A100)", "54.8 / 32.2"),
        ("rfdetrm768cos_cp_wtbd_group_full", RUNS, "WTBD-group, A1, s0", "51.5 / 30.7"),
        ("rfdetrm768cos_cp_wtbd_group_full_s1", RUNS, "WTBD-group, A1, s1", "58.5 / 32.7"),
        ("rfdetrm768cos_cp_d1_wtbd_group_full", RUNS, "WTBD-group, A2", "49.6 / 27.4"),
        ("rfdetrm768cos_cp_d2_wtbd_group_full", RUNS, "WTBD-group, A3", "55.3 / 33.2"),
        ("rfdetrm768cos_fp_wtbd_group_full", RUNS, "WTBD-group, FP, s0", "58.7 / 34.3"),
        ("rfdetrm768cos_fp_wtbd_group_full_s1", RUNS, "WTBD-group, FP, s1", "59.1 / 34.5"),
        ("rfdetrm640cos_wtblade_group_full", RUNS, "WTBlade-group, A0, s0", "38.5 / 16.2"),
        ("rfdetrm640cos_wtblade_group_full_s1", ARCH, "WTBlade-group, A0, s1 (T4)", "41.5 / 16.9"),
        ("rfdetrm640cos_wtblade_group_full_s1", RUNS, "WTBlade-group, A0, s1 (A100)", "41.0 / 16.9"),
        ("rfdetrm640cos_cp_wtblade_group_full", RUNS, "WTBlade-group, A1, s0", "41.8 / 17.3"),
        ("rfdetrm640cos_cp_wtblade_group_full_s1", RUNS, "WTBlade-group, A1, s1", "39.3 / 16.5"),
        ("rfdetrm640cos_cp_d1_wtblade_group_full", RUNS, "WTBlade-group, A2", "40.1 / 16.1"),
        ("rfdetrm640cos_cp_d2_wtblade_group_full", RUNS, "WTBlade-group, A3", "41.7 / 16.6"),
        ("rfdetrm640cos_fp_wtblade_group_full", RUNS, "WTBlade-group, FP, s0", "41.6 / 17.3"),
        ("rfdetrm640cos_fp_wtblade_group_full_s1", RUNS, "WTBlade-group, FP, s1", "40.5 / 17.2"),
        ("rfdetrm768cos_wtbd_official", RUNS, "WTBD-official, G26", o("WTBD-official", "rfdetrm768cos_wtbd_official")),
        ("rfdetrm768cos_fp_wtbd_official", RUNS, "WTBD-official, FP", o("WTBD-official", "rfdetrm768cos_fp_wtbd_official")),
        ("rfdetrm640cos_windsurface_official", RUNS, "Wind Surface, G26", o("Wind Surface Defect", "rfdetrm640cos_windsurface_official")),
        ("rfdetrm640cos_fp_windsurface_official", RUNS, "Wind Surface, FP", o("Wind Surface Defect", "rfdetrm640cos_fp_windsurface_official")),
        ("rfdetrm640cos_wtblade_defect_official", RUNS, "WTBlade-Defect, G26", o("WTBlade-Defect", "rfdetrm640cos_wtblade_defect_official")),
        ("rfdetrmfp_wtbd_group_f25", RUNS, "WTBD-group 25\\%, class-matched, FP", "-- / 24.6"),
        ("rfdetrmfp_rand1_wtbd_group_f25", RUNS, "WTBD-group 25\\%, rand1, FP", "-- / 20.4"),
        ("rfdetrmfp_rand2_wtbd_group_f25", RUNS, "WTBD-group 25\\%, rand2, FP", "-- / 22.1"),
        ("rfdetrmfp_rand3_wtbd_group_f25", RUNS, "WTBD-group 25\\%, rand3, FP", "-- / 14.1"),
        ("rfdetrm640ag_id_wtbd_group", RUNS, "ID-WTBD, class-agnostic", "72.9 / 41.9"),
        ("rfdetrm640ag_id_wtblade_group", RUNS, "ID-WTBlade, class-agnostic", "51.7 / 21.5"),
        ("rfdetrm640ag_x_wtbd_group", RUNS, "X-WTBD, class-agnostic (own val)", "17.7 / 6.0"),
        ("rfdetrm640ag_x_wtblade_group", RUNS, "X-WTBlade, class-agnostic (own val)", "2.4 / 0.5"),
    ]
    rows = []
    for run, base, lab, test in spec:
        path = os.path.join(base, run)
        if not os.path.exists(os.path.join(path, "metrics.csv")):
            print("missing", path)
            continue
        n, e, v50, v95 = val_best(path)
        rows.append(f"\\texttt{{{run.replace('_', chr(92) + '_')}}} & {lab} & {n} ({e}) & {v50:.1f} / {v95:.1f} & {test} \\\\")
    rows.append("\\texttt{yolo12l\\_wtblade\\_group} & WTBlade-group, YOLO12-L & 50 (43) & 33.6 / 12.5 & 32.2 / 13.3 \\\\")
    return ("{\\scriptsize\n\\begin{longtable}{@{}l l l l l@{}}\n"
            "\\caption{Runs of the final phase (G26--G29, G28, YOLO12-L). Val best is the best EMA epoch on validation (Ultralytics' best "
            "epoch for YOLO12-L), with the epoch in parentheses; test values are those of the tables in the main text (COCO-style "
            "evaluator, percent). On the 25\\% runs only \\mcoco was scored per subset.}\\label{tab:newruns}\\\\\n\\toprule\n"
            "Run & Data, arm & Epochs (best) & Val best \\mfifty / \\mcoco & Test \\mfifty / \\mcoco \\\\\n\\midrule\n\\endfirsthead\n"
            "\\toprule Run & Data, arm & Epochs (best) & Val best & Test \\\\\\midrule\n\\endhead\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n}\n")


def main():
    for name, f in (("gen_audit", audit), ("gen_registry", registry), ("gen_newruns", final_runs)):
        p = os.path.join(OUT, f"{name}.tex")
        open(p, "w", encoding="utf-8").write("% generated by tools/build_ieee_tables.py; do not edit\n" + f())
        print("wrote", p)


if __name__ == "__main__":
    main()
