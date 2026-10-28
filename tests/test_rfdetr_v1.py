"""V1 pieces (tools/rfdetr_v1.py): the subset copy keeps only subset images in train/ and renames val to valid/; boxes
are rescaled to 640-input pixels; imported predictions land under local absolute paths."""

import json
import os
import shutil
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rfdetr_v1 as v1  # noqa: E402

TMP = os.path.join(ROOT, "runs", "tests", "rfdetr_v1")


def fixture():
    shutil.rmtree(TMP, ignore_errors=True)
    src = os.path.join(TMP, "src")
    for split, files in [("train", ["a.jpg", "b.jpg", "c.jpg"]), ("val", ["v.jpg"]), ("test", ["t1.jpg", "t2.jpg"])]:
        for d in ("images", "labels"):
            os.makedirs(os.path.join(src, split, d))
        for f in files:
            open(os.path.join(src, split, "images", f), "w").write("x")
            open(os.path.join(src, split, "labels", f[:-4] + ".txt"), "w").write("4 0.5 0.5 0.1 0.2\n")
    yaml.safe_dump(dict(nc=6, names=["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]),
                   open(os.path.join(src, "data.yaml"), "w"))
    open(os.path.join(TMP, "subset.txt"), "w").write("a.jpg\nc.jpg\n")
    return src


def test_prepare():
    src = fixture()
    out = os.path.join(TMP, "ds")
    v1.prepare(src, os.path.join(TMP, "subset.txt"), out)
    assert sorted(os.listdir(os.path.join(out, "train", "images"))) == ["a.jpg", "c.jpg"]
    assert sorted(os.listdir(os.path.join(out, "train", "labels"))) == ["a.txt", "c.txt"]
    assert os.listdir(os.path.join(out, "valid", "images")) == ["v.jpg"]
    assert len(os.listdir(os.path.join(out, "test", "images"))) == 2
    assert not any(os.path.islink(os.path.join(out, "train", "images", f)) for f in ("a.jpg", "c.jpg"))
    y = yaml.safe_load(open(os.path.join(out, "data.yaml")))
    assert y["names"][4] == "crack" and y["val"] == "valid/images"
    print("ok test_prepare")


def test_rows_and_import():
    r = v1.to_rows([[10, 20, 110, 220]], [0.9], [4], 640 / 1280)
    assert np.allclose(r, [[5, 10, 55, 110, 0.9, 4]])
    src = os.path.join(TMP, "src")
    run = os.path.join(TMP, "rfdetrm_x")
    os.makedirs(run, exist_ok=True)
    json.dump({"t1.jpg": r.tolist(), "t2.jpg": []}, open(os.path.join(run, "preds_test.json"), "w"))
    cache = {"test": os.path.join(TMP, "cache_test"), "val": os.path.join(TMP, "cache_val")}
    v1.import_preds(run, data=src, cache=cache)
    p = json.load(open(os.path.join(cache["test"], "rfdetrm_x.json")))
    assert p[os.path.abspath(os.path.join(src, "test", "images", "t1.jpg"))] == r.tolist()
    assert not os.path.exists(os.path.join(cache["val"], "rfdetrm_x.json"))  # no preds_val.json: skipped
    print("ok test_rows_and_import")


if __name__ == "__main__":
    test_prepare()
    test_rows_and_import()
