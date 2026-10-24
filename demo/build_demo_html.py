"""Dựng trang demo offline RPDI-EE.

Chạy từ thư mục gốc repo (cần `pip install tokenizers` và tokenizer.json của DeepSeek-R1-Distill-Qwen-1.5B):
    python demo/prep_demo_data.py . path/to/tokenizer.json demo/demo_data.json
    python demo/build_demo_html.py demo/demo_data.json demo/rpdi_demo_offline.html
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
data_path, out_path = sys.argv[1], sys.argv[2]

d = json.load(open(data_path, encoding="utf-8"))
d.pop("checks", None)
# Thoát '</' để không đóng nhầm thẻ <script>; thoát U+FFFD (mảnh UTF-8 khi giải mã từng token)
data = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("\ufffd", "\\ufffd")
sim = open(os.path.join(here, "sim.js"), encoding="utf-8").read()
sim = sim.replace('if (typeof module !== "undefined") module.exports = { simulateRpdi, rpdiSeries };\n', "")
page = open(os.path.join(here, "template.html"), encoding="utf-8").read().replace("__DATA__", data).replace("__SIM__", sim)

with open(out_path, "w", encoding="utf-8") as f:
    f.write('<!doctype html>\n<html lang="vi"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "<style>body{margin:0}[hidden]{display:none!important}</style>\n</head><body>\n")
    f.write(page)
    f.write("\n</body></html>\n")
print("Đã ghi", out_path, f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
