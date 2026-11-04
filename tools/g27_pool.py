"""G27 (research_log.md): the extra-data pools for directions 1 (in-domain self-supervised adaptation of RF-DETR's
encoder) and 2 (multi-dataset supervised stage 1 with one class slot per dataset class).

Every local blade dataset contributes its labelled images with its own official classes, unless the image is an
evaluation image of the protocols the pool serves, or shares content with one. Two pools:

  L (leak-free protocols; scene-level clean)  evaluation = WTBD-group val + test, WTBlade-group val + test.
      Scene rule: WTBD images only from WTBD-group train; WTBlade-Defect files only from scene groups assigned to
      WTBlade-group train (runs/analysis/wtblade/split_groups.csv). Then the twin rule below for every source.
  O (official protocols; image-level clean, as the protocols themselves are)  evaluation = WTBD-official val + test,
      WTBlade-Defect valid + test, Wind Surface val (= its test) + the seeded 10% carve of its train. Twin rule only.

Twin rule (the leak-audit tools, tools/matching.py), fixed before any G27 run:
  1. same file (dataset, file name; WTBD-group and WTBlade-group reuse the source file names);
  2. DINOv2 ViT-S/14 cosine >= 0.90 to an evaluation image (0.93 = matching.DINO_GROUP is the group splits' "same
     scene"; the pool keeps a margin below it, see TWIN_COS);
  3. ORB/RANSAC >= 25 inliers (matching.MIN_INLIERS, flips included) among each evaluation image's 30 nearest pool
     images with cosine >= 0.70, and among each pool image's 3 nearest evaluation images with cosine >= 0.80;
  4. pixel twin: correlation >= 0.97 of 48-px grey images over the 8 flips/rotations (twin test of
     tools/ws_pixel_twins.py at 0.99, with a margin);
  5. closure: a pool image within cosine >= 0.90 or pixel correlation >= 0.97 of any excluded or evaluation image
     is excluded too; 3-5 repeat until nothing changes.
(History: a first all-protocol pool with rules 1, 2, 3a and DINOv2 closure kept 830 pixel twins, found by `audit`;
with rule 4 added, the audit's closest remaining pairs were still neighbouring frames and re-exposed copies of the
same scenes, hence the scene rule of pool L and rule 3b. Archived: runs/analysis/g27/g27_pool_v1_dino_orb_only.csv.)
WTBs2025 GAN images (names containing "GAN") are dropped (synthetic). Images without boxes are dropped: the pool is
the same image set for direction 1 (images only) and direction 2 (images + labels).

  .venv/bin/python tools/g27_pool.py build --pool L   -> tools/splits/g27_pool_L.csv, runs/analysis/g27/pool_L.md
  .venv/bin/python tools/g27_pool.py audit --pool L   -> runs/analysis/g27/pool_L_audit.md (+ contact sheets)
  .venv/bin/python tools/g27_pool.py export --pool L --out g27-pool-L
      -> {train,valid,test}/{images,labels}, data.yaml (slot names "<dataset>:<class>"), slots.json, g27_pool_L.csv
         train = the kept pool; valid = test = WTBD-group val + WTBlade-group val in their slots (logging only)
Upload the export folder as a Kaggle dataset (contains DTU-derived WTBlade frames, CC BY-NC: research use only).
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matching  # noqa: E402

POOLS = ("L", "O")
csv_path = lambda pool: os.path.join(ROOT, "tools", "splits", f"g27_pool_{pool}.csv")  # noqa: E731
report_path = lambda pool: os.path.join(ROOT, "runs", "analysis", "g27", f"pool_{pool}.md")  # noqa: E731
WTBLADE_GROUPS = os.path.join(ROOT, "runs", "analysis", "wtblade", "split_groups.csv")
POOL_K, POOL_COS = 3, 0.80
CACHE = os.path.join(ROOT, ".dataset_cache", "dinov2_vits14.npz")
# exclusion thresholds sit below the leak-audit thresholds (DINOv2 0.93 = matching.DINO_GROUP, pixel 0.99) by a
# margin: at those exact values the pool L audit's closest kept pairs were still same-scene (a re-exposed WTBD copy at
# pixel 0.98, re-tinted and neighbouring drone frames at cosine 0.92-0.93)
TWIN_COS, CAND_COS, CAND_K, PIXEL_TWIN = 0.90, 0.70, 30, 0.97
AUDIT_COS, AUDIT_PIX = matching.DINO_GROUP, 0.99
WTBS = "WTBs2025"
# slot order = this order; (root, splits) relative to the repo
SOURCES = {"wtbd": ("external/WTBD-official", ["train", "val", "test"]),
           "wtblade": ("WTBlade-Defect", ["train", "valid", "test"]),
           "ws": ("WindSurface-Defect", ["train", "val"]),
           "bj": ("wind-turbine.v1i.yolov12", ["train", "valid", "test"]),
           "wtbs": (WTBS, None)}  # one folder per class, each with images/ labels/
FIELDS = ["path", "source", "split", "name", "boxes", "status", "match", "cos"]


def names_of(root):
    n = yaml.safe_load(open(os.path.join(ROOT, root, "data.yaml")))["names"]
    return list(n.values()) if isinstance(n, dict) else list(n)


def label_path(img):
    d, f = os.path.split(img)
    return os.path.join(os.path.dirname(d), "labels", os.path.splitext(f)[0] + ".txt")


def read_rows(img):
    p = os.path.join(ROOT, label_path(img))
    rows = [l.split() for l in open(p) if l.strip()] if os.path.exists(p) else []
    return [r for r in rows if len(r) == 5]  # boxes only (no polygons in these exports)


def records():
    """All source images: dicts with path (repo-relative), source, split, name."""
    out = []
    for src, (root, splits) in SOURCES.items():
        if splits is None:
            splits = sorted(d for d in os.listdir(os.path.join(ROOT, root)) if os.path.isdir(os.path.join(ROOT, root, d, "images")))
        for s in splits:
            d = os.path.join(root, s, "images")
            for f in sorted(os.listdir(os.path.join(ROOT, d))):
                out.append(dict(path=os.path.join(d, f), source=src, split=s, name=f))
    return out


def ws_carve(seed=0, frac=0.1):
    """The Wind Surface validation images of rfdetr_v1.prepare_official (--carve 0.1, seed 0)."""
    train = sorted(os.listdir(os.path.join(ROOT, "WindSurface-Defect", "train", "images")))
    return set(random.Random(seed).sample(train, round(frac * len(train))))


def eval_keys(pool):
    """(source, file name) of every evaluation image of the protocols the pool serves."""
    ls = lambda d: os.listdir(os.path.join(ROOT, d))  # noqa: E731
    if pool == "L":
        wtbd = ("external/WTBD-group/val/images", "external/WTBD-group/test/images")
        wtbl = ("external/WTBlade-group/val/images", "external/WTBlade-group/test/images")
        ws = set()
    else:
        wtbd = ("external/WTBD-official/val/images", "external/WTBD-official/test/images")
        wtbl = ("WTBlade-Defect/valid/images", "WTBlade-Defect/test/images")
        ws = set(ls("WindSurface-Defect/val/images")) | ws_carve()
    keys = {("wtbd", f) for d in wtbd for f in ls(d)} | {("wtblade", f) for d in wtbl for f in ls(d)}
    return keys | {("ws", f) for f in ws}


def scene_rule(recs):
    """Pool L: WTBD only from WTBD-group train; WTBlade-Defect only from scene groups of WTBlade-group train."""
    wtbd_train = set(os.listdir(os.path.join(ROOT, "external", "WTBD-group", "train", "images")))
    sub = {r["image"]: r["subset"] for r in csv.DictReader(open(WTBLADE_GROUPS))}
    for r in recs:
        if r["status"]:
            continue
        if r["source"] == "wtbd" and r["name"] not in wtbd_train:
            r["status"] = "scene"
        elif r["source"] == "wtblade" and sub.get(r["path"]) != "train":
            r["status"] = "scene" if r["path"] in sub else "no_group"


def build(pool, workers=8):
    recs = records()
    ev = eval_keys(pool)
    for r in recs:
        r["boxes"] = len(read_rows(r["path"]))
        r["status"], r["match"], r["cos"] = "", "", ""
    by_key = {}
    for i, r in enumerate(recs):
        by_key.setdefault((r["source"], r["name"]), []).append(i)
    missing = [k for k in ev if k not in by_key]
    assert not missing, f"{len(missing)} evaluation images not found among the sources, e.g. {sorted(missing)[:3]}"
    E_idx = sorted({by_key[k][0] for k in ev})  # one copy per evaluation image
    for k in ev:
        for i in by_key[k]:
            recs[i]["status"] = "eval"
    if pool == "L":
        scene_rule(recs)
    for r in recs:
        if not r["status"] and r["source"] == "wtbs" and "GAN" in r["name"]:
            r["status"] = "gan"
        elif not r["status"] and r["boxes"] == 0:
            r["status"] = "no_boxes"
    cwd = os.getcwd()
    os.chdir(ROOT)  # cache keys are repo-relative paths
    try:
        emb = matching.dinov2_embeddings([r["path"] for r in recs], CACHE)
    finally:
        os.chdir(cwd)
    cand = [i for i, r in enumerate(recs) if not r["status"]]
    print(f"pool {pool}: {len(E_idx)} evaluation images, {len(cand)} candidates", flush=True)
    C, E = emb[cand], emb[E_idx]
    twin = {}  # record index -> (status, matched path, score)
    orb = []  # (evaluation record, candidate record, cosine)
    for a in range(0, len(E), 512):
        s = E[a:a + 512] @ C.T
        for j, row in enumerate(s):
            e = E_idx[a + j]
            for c in np.flatnonzero(row >= TWIN_COS):
                twin.setdefault(cand[c], ("twin_dino", recs[e]["path"], float(row[c])))
            orb += [(e, cand[c], float(row[c])) for c in np.argsort(-row)[:CAND_K] if CAND_COS <= row[c] < TWIN_COS]
    G = np.stack([unit(gray48(recs[i]["path"])[None])[0] for i in cand]).astype(np.float32)  # (n_cand, 2304)
    evset, done = set(E_idx), set()
    frontier = list(E_idx) + list(twin)
    while True:
        orb = [(e, i, c) for e, i, c in orb if i not in twin and (e, i) not in done]
        if orb:
            print(f"  ORB: verifying {len(orb)} pairs", flush=True)
            todo = [(e, i) for e, i, _ in orb]
            done |= set(todo)
            ver = matching.verify_pairs({k: os.path.join(ROOT, recs[k]["path"]) for p in todo for k in p}, todo, workers)
            cos_of = {(e, i): c for e, i, c in orb}
            for (e, i), (inl, _) in ver.items():
                if inl >= matching.MIN_INLIERS and i not in twin:
                    twin[i] = ("twin_orb", recs[e]["path"], cos_of[(e, i)])
                    frontier.append(i)
        n0 = len(twin)
        while frontier:  # pixel twins and closure
            new, hits = [], {}
            F = emb[frontier]
            V = np.stack([gray48(recs[i]["path"]) for i in frontier])
            for a in range(0, len(F), 512):
                s = F[a:a + 512] @ C.T
                for j, c in zip(*np.nonzero(s >= TWIN_COS)):
                    hits.setdefault(c, (a + j, float(s[j, c]), "dino"))
            for v in dihedral(V):
                Vn = unit(np.ascontiguousarray(v))
                for a in range(0, len(Vn), 512):
                    s = Vn[a:a + 512] @ G.T
                    for j, c in zip(*np.nonzero(s >= PIXEL_TWIN)):
                        hits.setdefault(c, (a + j, float(s[j, c]), "pixel"))
            for c, (j, sc, how) in hits.items():
                i, f = cand[c], frontier[j]
                if i in twin:
                    continue
                twin[i] = (("twin_pixel" if how == "pixel" else "twin_dino") if f in evset else "closure", recs[f]["path"], sc)
                new.append(i)
            frontier = new
            print(f"  pixel twins / closure +{len(new)}", flush=True)
        # pool side: each remaining candidate's nearest evaluation images go to ORB
        keep = [c for c, i in enumerate(cand) if i not in twin]
        orb = []
        for a in range(0, len(keep), 4096):
            s = C[keep[a:a + 4096]] @ E.T
            for j, row in enumerate(s):
                i = cand[keep[a + j]]
                orb += [(E_idx[k], i, float(row[k])) for k in np.argsort(-row)[:POOL_K] if row[k] >= POOL_COS
                        and (E_idx[k], i) not in done]
        if not orb and len(twin) == n0:
            break
    for i, (st, m, c) in twin.items():
        recs[i].update(status=st, match=m, cos=f"{c:.4f}")
    for r in recs:
        r["status"] = r["status"] or "kept"
    os.makedirs(os.path.dirname(csv_path(pool)), exist_ok=True)
    with open(csv_path(pool), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in recs)
    report(recs, pool)


def report(recs, pool):
    sts = ["kept", "eval", "scene", "no_group", "twin_dino", "twin_orb", "twin_pixel", "closure", "no_boxes", "gan"]
    lines = [f"# G27 extra-data pool {pool} (tools/g27_pool.py)", "",
             f"Twin rule: same file; DINOv2 cosine >= {TWIN_COS}; ORB/RANSAC >= {matching.MIN_INLIERS} inliers among each "
             f"evaluation image's {CAND_K} nearest pool images with cosine >= {CAND_COS} and each pool image's {POOL_K} nearest "
             f"evaluation images with cosine >= {POOL_COS}; pixel twin >= {PIXEL_TWIN} (48 px grey, 8 flips/rotations); closure at cosine >= {TWIN_COS} or pixel >= {PIXEL_TWIN}.", "",
             "| source | files | " + " | ".join(sts) + " | kept boxes |", "|---|---|" + "---|" * (len(sts) + 1)]
    for src in list(SOURCES) + ["all"]:
        rs = [r for r in recs if src == "all" or r["source"] == src]
        cnt = {s: sum(r["status"] == s for r in rs) for s in sts}
        kb = sum(int(r["boxes"]) for r in rs if r["status"] == "kept")
        lines.append(f"| {src} | {len(rs)} | " + " | ".join(str(cnt[s]) for s in sts) + f" | {kb} |")
    lines += ["", "Kept boxes per class (slot):", "", "| slot | boxes | images |", "|---|---|---|"]
    for src, (root, _) in SOURCES.items():
        nm = names_of(root)
        rs = [r for r in recs if r["source"] == src and r["status"] == "kept"]
        cnt, img = np.zeros(len(nm), int), np.zeros(len(nm), int)
        for r in rs:
            cl = [int(x[0]) for x in read_rows(r["path"])]
            for c in cl:
                cnt[c] += 1
            for c in set(cl):
                img[c] += 1
        lines += [f"| {src}:{n} | {cnt[k]} | {img[k]} |" for k, n in enumerate(nm)]
    os.makedirs(os.path.dirname(report_path(pool)), exist_ok=True)
    open(report_path(pool), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def slots():
    """{source: (offset, names)} in SOURCES order, and the flat slot names."""
    out, flat = {}, []
    for src, (root, _) in SOURCES.items():
        nm = names_of(root)
        out[src] = (len(flat), nm)
        flat += [f"{src}:{n}" for n in nm]
    return out, flat


def flat_name(r):
    return f"{r['source']}__{r['split'].replace(' ', '-')}__{r['name']}"


def write_item(src_img, rows, offset, out, split, name):
    shutil.copy2(src_img, os.path.join(out, split, "images", name))
    with open(os.path.join(out, split, "labels", os.path.splitext(name)[0] + ".txt"), "w") as f:
        f.write("".join(f"{int(r[0]) + offset} {' '.join(r[1:])}\n" for r in rows))


def export(pool, out):
    sl, flat = slots()
    recs = list(csv.DictReader(open(csv_path(pool))))
    for s in ("train", "valid", "test"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(out, s, d), exist_ok=True)
    kept = [r for r in recs if r["status"] == "kept"]
    for n, r in enumerate(kept):
        write_item(os.path.join(ROOT, r["path"]), read_rows(r["path"]), sl[r["source"]][0], out, "train", flat_name(r))
        if n % 2000 == 0:
            print(f"  train {n}/{len(kept)}", flush=True)
    nval = 0
    for src, root in (("wtbd", "external/WTBD-group"), ("wtblade", "external/WTBlade-group")):
        d = os.path.join(root, "val", "images")
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            for s in ("valid", "test"):
                write_item(os.path.join(ROOT, d, f), read_rows(os.path.join(d, f)), sl[src][0], out, s, f"{src}__val__{f}")
            nval += 1
    yaml.safe_dump(dict(path=".", train="train/images", val="valid/images", test="test/images", nc=len(flat), names=flat),
                   open(os.path.join(out, "data.yaml"), "w"), sort_keys=False)
    json.dump({s: dict(offset=o, names=n) for s, (o, n) in sl.items()}, open(os.path.join(out, "slots.json"), "w"), indent=1)
    shutil.copy2(csv_path(pool), os.path.join(out, os.path.basename(csv_path(pool))))
    print(f"train {len(kept)}, valid/test {nval} images, {len(flat)} slots -> {out}")


def gray48(p, s=48):
    import cv2

    return cv2.resize(cv2.imread(os.path.join(ROOT, p), cv2.IMREAD_GRAYSCALE), (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)


def dihedral(V):
    """The 8 flips/rotations of a stack of square images (n, s, s)."""
    T = np.transpose(V, (0, 2, 1))
    return [V, V[:, :, ::-1], V[:, ::-1, :], V[:, ::-1, ::-1], np.rot90(V, 1, (1, 2)), np.rot90(V, 3, (1, 2)), T,
            np.rot90(T, 2, (1, 2))]


def unit(a):
    a = a.reshape(len(a), -1)
    a = a - a.mean(1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def audit(pool):
    """Independent check of the kept pool: best pixel correlation (48 px grey, 8 flips/rotations; the Wind Surface
    twin test of tools/ws_pixel_twins.py, >= 0.99 = pixel twin) and best DINOv2 cosine of every kept image against
    every evaluation image, plus a contact sheet of the 12 closest pairs by each measure."""
    import cv2

    recs = list(csv.DictReader(open(csv_path(pool))))
    ev = eval_keys(pool)
    kept = [r["path"] for r in recs if r["status"] == "kept"]
    seen, evp = set(), []
    for r in recs:
        if (r["source"], r["name"]) in ev and (r["source"], r["name"]) not in seen:
            seen.add((r["source"], r["name"]))
            evp.append(r["path"])
    print(f"{len(kept)} kept vs {len(evp)} evaluation images", flush=True)
    K = unit(np.stack([gray48(p) for p in kept]))
    V = np.stack([gray48(p) for p in evp])
    best, arg = np.full(len(kept), -1.0, np.float32), np.zeros(len(kept), int)
    for v in dihedral(V):
        Vn = unit(np.ascontiguousarray(v))
        for a in range(0, len(K), 4096):
            s = K[a:a + 4096] @ Vn.T
            j = s.argmax(1)
            m = s[np.arange(len(s)), j]
            upd = m > best[a:a + 4096]
            best[a:a + 4096][upd], arg[a:a + 4096][upd] = m[upd], j[upd]
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        Ek, Ee = matching.dinov2_embeddings(kept, CACHE), matching.dinov2_embeddings(evp, CACHE)
    finally:
        os.chdir(cwd)
    cos = np.full(len(kept), -1.0, np.float32)
    carg = np.zeros(len(kept), int)
    for a in range(0, len(Ek), 4096):
        s = Ek[a:a + 4096] @ Ee.T
        carg[a:a + 4096] = s.argmax(1)
        cos[a:a + 4096] = s.max(1)
    out = os.path.dirname(report_path(pool))
    lines = [f"# G27 pool {pool} audit: kept pool vs every evaluation image", "",
             f"{len(kept)} kept images, {len(evp)} evaluation images.", "",
             "| measure | max | 99.9th pct | 99th pct | median | kept images >= twin threshold |", "|---|---|---|---|---|---|",
             f"| pixel corr, 48 px, 8 flips/rots (audit twin >= {AUDIT_PIX}; rule < {PIXEL_TWIN}) | {best.max():.3f} | "
             f"{np.percentile(best, 99.9):.3f} | {np.percentile(best, 99):.3f} | {np.median(best):.3f} | {(best >= AUDIT_PIX).sum()} |",
             f"| DINOv2 cosine (audit twin >= {AUDIT_COS}; rule < {TWIN_COS}) | {cos.max():.3f} | {np.percentile(cos, 99.9):.3f} | "
             f"{np.percentile(cos, 99):.3f} | {np.median(cos):.3f} | {(cos >= AUDIT_COS).sum()} |", "",
             f"Closest pairs (contact sheets `pool_{pool}_audit_pixel.jpg`, `pool_{pool}_audit_dino.jpg`; kept | evaluation):", ""]
    for tag, sc, ag in (("pixel", best, arg), ("dino", cos, carg)):
        top = np.argsort(-sc)[:12]
        tiles = []
        for i in top:
            lines.append(f"- {tag} {sc[i]:.3f}: `{kept[i]}` | `{evp[ag[i]]}`")
            pair = [cv2.resize(cv2.imread(os.path.join(ROOT, p)), (200, 200)) for p in (kept[i], evp[ag[i]])]
            tiles.append(np.concatenate(pair + [np.full((200, 8, 3), 255, np.uint8)], 1))
        rows = [np.concatenate(tiles[k:k + 3], 1) for k in range(0, len(tiles), 3)]
        cv2.imwrite(os.path.join(out, f"pool_{pool}_audit_{tag}.jpg"), np.concatenate(rows, 0))
    open(os.path.join(out, f"pool_{pool}_audit.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "audit", "export", "report"])
    ap.add_argument("--pool", required=True, choices=POOLS)
    ap.add_argument("--out", help="export: output folder")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.pool, args.workers)
    elif args.cmd == "audit":
        audit(args.pool)
    elif args.cmd == "export":
        export(args.pool, args.out)
    else:
        report(list(csv.DictReader(open(csv_path(args.pool)))), args.pool)


if __name__ == "__main__":
    main()
