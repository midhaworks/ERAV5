"""Inline site/data.json into site/app_template.html -> site/index.html (self-contained)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
data = open(os.path.join(H, "site", "data.json"), encoding="utf-8").read()
blob = json.dumps(json.loads(data), ensure_ascii=False).replace("</", "<\\/")
tpl = open(os.path.join(H, "app_template.html"), encoding="utf-8").read()
out = tpl.replace("/*__DATA__*/", blob)
open(os.path.join(H, "site", "index.html"), "w", encoding="utf-8").write(out)
print("wrote site/index.html  (%.0f KB)" % (len(out) / 1024))
