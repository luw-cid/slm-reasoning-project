"""Tải và chuẩn hóa SVAMP và MATH500 về cùng định dạng với gsm8k_test_full.json:
[{"id": int, "question": str, "ground_truth": str, ...}, ...]

    python common/prepare_datasets.py            # ghi common/data/svamp_full.json, math500_test.json

Nguồn:
- SVAMP (Patel et al., 2021): 1.000 bài, bản gốc của tác giả trên GitHub.
- MATH500 (Lightman et al., 2023): tập con 500 bài của MATH test, bản HuggingFaceH4/MATH-500.
"""
import json
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-prep"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def number_str(x) -> str:
    f = float(x)
    return str(int(f)) if f.is_integer() else repr(f)


def svamp_records(items):
    out = []
    for i, it in enumerate(items):
        body, q = it["Body"].strip(), it["Question"].strip()
        out.append({"id": i, "question": f"{body} {q}",  # Body + " " + Question, as in the SVAMP release
                    "ground_truth": number_str(it["Answer"]), "source_id": it.get("ID"),
                    "type": it.get("Type")})
    return out


def math500_records(lines):
    out = []
    for i, line in enumerate(x for x in lines if x.strip()):
        it = json.loads(line)
        out.append({"id": i, "question": it["problem"], "ground_truth": it["answer"],
                    "subject": it.get("subject"), "level": it.get("level"), "unique_id": it.get("unique_id")})
    return out


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    jobs = [
        ("svamp_full.json", lambda: svamp_records(json.loads(fetch(SVAMP_URL)))),
        ("math500_test.json", lambda: math500_records(fetch(MATH500_URL).splitlines())),
    ]
    for name, build in jobs:
        path = os.path.join(DATA_DIR, name)
        recs = build()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        print(f"{name}: {len(recs)} bài -> {path}")


if __name__ == "__main__":
    main()
