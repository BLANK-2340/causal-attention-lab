"""Strict inference checkpoints and reproducible evidence for tiny_lm.py.

JSON only, no pickle or executable deserialization. Checksums detect accidental
corruption, NOT malicious authorship. Checkpoints omit Adam state and cannot
resume training. Format v1 deliberately accepts only the bounded public model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tiny_lm as lm

ROOT = Path(__file__).resolve().parent
LIMIT = 128_000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_hash():
    return hashlib.sha256((ROOT / "tiny_lm.py").read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    with open(path, "rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError("file exceeds bounded format limit")
    return json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_pairs)


def pack(model):
    payload = {"format": 1, "source_sha256": source_hash(), "vocabulary": list(lm.VOCAB),
               "attention": model.attention, "parameters": {n: p.tolist() for n, p in model.p.items()}}
    return {"payload": payload, "sha256": hashlib.sha256(canonical(payload).encode()).hexdigest()}


def unpack(document):
    if not isinstance(document, dict) or set(document) != {"payload", "sha256"}:
        raise ValueError("invalid checkpoint envelope")
    p = document["payload"]
    if not isinstance(p, dict) or set(p) != {"format", "source_sha256", "vocabulary", "attention", "parameters"}:
        raise ValueError("invalid checkpoint payload")
    if (type(p["format"]) is not int or p["format"] != 1 or p["source_sha256"] != source_hash()
            or p["vocabulary"] != list(lm.VOCAB) or type(p["attention"]) is not bool):
        raise ValueError("unsupported version, source, vocabulary or attention flag")
    if hashlib.sha256(canonical(p).encode()).hexdigest() != document["sha256"]:
        raise ValueError("checksum mismatch")
    model = lm.Model(attention=p["attention"])
    if not isinstance(p["parameters"], dict) or set(p["parameters"]) != set(model.p):
        raise ValueError("incorrect parameter names")
    checked = {}
    for name, expected in model.p.items():
        value = p["parameters"][name]
        # Shape and numeric type checked before float conversion; strings/bools refused.
        raw = np.asarray(value)
        if (raw.shape != expected.shape or raw.dtype.kind not in "iuf"
                or any(type(cell) not in (int, float) for cell in np.asarray(value, dtype=object).flat)):
            raise ValueError("invalid parameter shape/type: " + name)
        array = raw.astype(np.float64)
        if not np.isfinite(array).all() or np.max(np.abs(array)) > 100:
            raise ValueError("nonfinite/out-of-format parameter: " + name)
        checked[name] = array.copy()
    model.p = checked
    return model


def load(path):
    return unpack(read_json(path))


def write_new(path, value):
    # Validate serialization BEFORE opening a destination; never overwrite.
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(encoded)


def save(path, model):
    document = pack(model)
    unpack(document)  # same validation contract as load, before file creation
    write_new(path, document)


def evidence():
    return {"source_sha256": source_hash(), "study": lm.experiment()}


def compare(actual, expected, path="root"):
    """Structure exact, numbers within 1e-10, runtime-version strings recorded only."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("evidence keys changed at " + path)
        for key in actual:
            if key not in ("python", "numpy"):
                compare(actual[key], expected[key], path + "." + key)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("evidence length changed at " + path)
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, path + f"[{i}]")
    elif type(actual) in (int, float) and type(expected) in (int, float):
        if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > 1e-10:
            raise ValueError("evidence number changed at " + path)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("evidence value changed at " + path)


def inspect(model, prompt):
    words = prompt.split()
    if (any(w not in lm.TOKEN or w in ("<bos>", "<eos>") for w in words)
            or len(words) >= lm.CONTEXT):
        raise ValueError("known-word prompt must fit the context without BOS/EOS")
    ids = [lm.TOKEN["<bos>"]] + [lm.TOKEN[w] for w in words]
    logits, cache = model.forward(np.array([ids], dtype=np.int64))
    probabilities = lm.softmax(logits[0, -1])
    return {"mode": "live CPU inference from loaded checkpoint, no retraining",
            "next_token_probability": {w: float(probabilities[i]) for i, w in enumerate(lm.VOCAB)},
            "attention_weights": [float(v) for v in cache[5][0, -1]] if model.attention else None,
            "input_tokens": [lm.VOCAB[i] for i in ids],
            "caution": "Attention weights are not causal explanations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "generate", "inspect", "export", "check"))
    parser.add_argument("path")
    parser.add_argument("--prompt", default="the red robot")
    args = parser.parse_args()
    if args.command == "train":
        model, _ = lm.fit(seed=7)
        save(args.path, model)
        print("Saved inference-only seed-7 checkpoint; existing files are never overwritten.")
    elif args.command == "generate":
        print(lm.generate(load(args.path), prompt=args.prompt))
    elif args.command == "inspect":
        print(json.dumps(inspect(load(args.path), args.prompt), indent=2, allow_nan=False))
    elif args.command == "export":
        write_new(args.path, evidence())
    else:
        compare(evidence(), read_json(args.path))
        print("Evidence matches source and recomputed six-run study (numeric tolerance 1e-10).")


if __name__ == "__main__":
    main()
