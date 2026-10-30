#!/usr/bin/env python3
"""
names_util.py — class-name handling (the original code never used class names).

Names can come from (in priority order):
  1. an explicit spec: path to .json / .yaml / .txt, or a comma-separated string
  2. a sidecar  <engine-name-without-precision>.names.json  written by export_engines.py
  3. the .pt checkpoint (Ultralytics)
  4. the ONNX metadata ('names' key written by Ultralytics export)
"""

import ast
import json
import os
import re


def norm_name(s):
    """'Aluminium Foil' / 'aluminium_foil' / 'aluminium-foil' -> 'aluminiumfoil'."""
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())


def _to_dict(obj):
    if isinstance(obj, dict):
        try:
            return {int(k): str(v) for k, v in obj.items()}
        except (TypeError, ValueError):
            raise ValueError("class-name dict keys must be integers (0, 1, 2, ...)")
    if isinstance(obj, (list, tuple)):
        return {i: str(v) for i, v in enumerate(obj)}
    raise ValueError(f"cannot interpret class names of type {type(obj).__name__}")


def parse_names(spec):
    """spec -> {int: str} or None."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, (dict, list, tuple)):
        return _to_dict(spec)
    spec = str(spec)
    looks_like_path = "," not in spec and (
        spec.lower().endswith((".json", ".yaml", ".yml", ".txt")) or "/" in spec or "\\" in spec)
    if looks_like_path and not os.path.isfile(spec):
        raise FileNotFoundError(f"class-names file not found: {spec}")
    if os.path.isfile(spec):
        ext = os.path.splitext(spec)[1].lower()
        with open(spec, "r", encoding="utf-8") as f:
            if ext == ".json":
                return _to_dict(json.load(f))
            if ext in (".yaml", ".yml"):
                import yaml
                d = yaml.safe_load(f)
                return _to_dict(d["names"] if isinstance(d, dict) and "names" in d else d)
            return _to_dict([ln.strip() for ln in f if ln.strip()])      # one name per line
    return _to_dict([s.strip() for s in spec.split(",") if s.strip()])


def names_from_pt(pt_path):
    from ultralytics import YOLO
    return _to_dict(dict(YOLO(pt_path).names))


def names_from_onnx(onnx_path):
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    for p in m.metadata_props:
        if p.key == "names":
            return _to_dict(ast.literal_eval(p.value))
    return None


def engine_base(engine_path):
    """'engines/best_fp16.engine' -> ('engines', 'best')."""
    d = os.path.dirname(engine_path) or "."
    stem = os.path.splitext(os.path.basename(engine_path))[0]
    return d, re.sub(r"_(fp32|fp16|int8)$", "", stem)


def resolve_names(spec=None, engine_path=None, pt_path=None):
    """Returns (names_dict_or_None, human-readable source)."""
    if spec:
        return parse_names(spec), f"explicit ({spec})"
    if engine_path:
        d, base = engine_base(engine_path)
        side = os.path.join(d, base + ".names.json")
        if os.path.isfile(side):
            return parse_names(side), f"sidecar ({side})"
    if pt_path and os.path.isfile(pt_path):
        try:
            return names_from_pt(pt_path), f"checkpoint ({pt_path})"
        except Exception as e:                                          # noqa: BLE001
            print(f"  [names] could not read names from {pt_path}: {e}")
    if engine_path:
        d, base = engine_base(engine_path)
        onnx_path = os.path.join(d, base + ".onnx")
        if os.path.isfile(onnx_path):
            try:
                n = names_from_onnx(onnx_path)
                if n:
                    return n, f"onnx metadata ({onnx_path})"
            except Exception as e:                                      # noqa: BLE001
                print(f"  [names] could not read names from {onnx_path}: {e}")
    return None, "not available"


def build_class_map(model_names, gt_cat_names):
    """
    Match model class indices to ground-truth category ids by NAME.
      model_names : {idx: name}        gt_cat_names : {cat_id: name}
    Returns (map {model_idx: cat_id}, report dict).
    """
    gt_by_norm = {}
    for cid, nm in gt_cat_names.items():
        gt_by_norm.setdefault(norm_name(nm), cid)
    cmap, unmapped = {}, []
    for idx, nm in (model_names or {}).items():
        cid = gt_by_norm.get(norm_name(nm))
        if cid is None:
            unmapped.append(nm)
        else:
            cmap[idx] = cid
    covered = set(cmap.values())
    return cmap, {
        "matched": {model_names[i]: gt_cat_names[c] for i, c in cmap.items()},
        "model_classes_without_gt": unmapped,
        "gt_classes_not_predictable": [n for c, n in gt_cat_names.items() if c not in covered],
    }
