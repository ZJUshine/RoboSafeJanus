#!/usr/bin/env python3
"""资产链接(看板 assets 列,用户自己定,不审核)的参考:为一个组的每个物体(label)找候选资产,
供人工挑选后写进 ``task/robosafejanus_assets.json``。

来源只有两种(与 ``check_task_alignment.py`` 一致):RoboDojo 官方(本地 ``Assets/Object/RoboDojo``)和 SimReady
(SimReady Central 搜索)。另外,同一 label 在别的组里已经选过的命令列为 ``reuse``(SimReady 类目在多个组里必须用
同一个模型,见 ``ext_assets/integrate/simready_assets.py``)。

候选存在 ``data/pipeline/<group>/assets/candidates.json``,缩略图在同目录 ``thumbs/``,对照页 ``candidates.html``::

    python scripts/internal/asset_candidates.py search <group> <label> --query "metal baking tray" [--query ...]
                                                [--grasped] [--k 4] [--local_k 3]
    python scripts/internal/asset_candidates.py page <group>               # 重新生成 candidates.html
    python scripts/internal/asset_candidates.py show <group> [label]        # 终端里看候选
    python scripts/internal/asset_candidates.py pick <group> <label> <candidate_id>   # 写进 robosafejanus_assets.json
    python scripts/internal/asset_candidates.py drop <group> <label>        # 去掉这个 label 的候选和已选

``--grasped`` 表示该物体会被夹爪抓取:可夹宽度 = 水平 bbox 的 ``min(x, y)``,超过 7 cm 时 simready_assets.py
会等比缩小(``_sNNN`` 变体),这里提示缩放比例;RoboDojo 官方资产不会被自动缩放,超宽时标红。
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(os.environ.get("RSJ_PIPELINE_DIR", ROOT / "data" / "pipeline"))
ASSETS_JSON = ROOT / "task" / "robosafejanus_assets.json"
OBJECTS = ROOT / "Assets" / "Object" / "RoboDojo"
SEARCH = "https://nvidia-simready-central.hf.space/search_hybrid"
MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
HF_PATH = re.compile(r"hf://huggingface\.co/datasets/([^/]+/[^/]+)/resolve/([0-9a-f]+)/([^/]+)/(.+)")
ROBODOJO_REPO = "RoboDojo-Benchmark/RoboDojo"
SIMREADY_REPOS = (
    "nvidia/simready-assets",
    "PalatialSim/palatial-nv-asset-library",
    "ForgeCAD/simready-assets",
    "LGElectronics/simready-assets",
)
SECTIONS = ("Rigid", "Articulation", "Geometry")
MAX_WIDTH = 0.07  # 与 simready_assets.py --max_width 默认值一致
SCALE_STEP = 0.05


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def group_dir(group: str) -> Path:
    return STATE_DIR / group / "assets"


def load_cands(group: str) -> dict:
    path = group_dir(group) / "candidates.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"group": group, "labels": {}}


def save_cands(data: dict) -> None:
    d = group_dir(data["group"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "candidates.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_assets() -> dict:
    return json.loads(ASSETS_JSON.read_text(encoding="utf-8"))


def _http(url: str, body: dict | None = None, timeout: int = 90) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": "curl/8"} | ({"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------------------
# 尺寸 / 抓取宽度
# ---------------------------------------------------------------------------


def grasp_note(extents, grasped: bool, scalable: bool) -> tuple[str, bool]:
    """``(说明, 是否有问题)``。"""
    if not grasped or not extents:
        return "", False
    width = min(extents[0], extents[1])
    if width <= MAX_WIDTH:
        return f"可夹宽度 {width * 100:.1f} cm", False
    if extents[2] <= MAX_WIDTH:
        # 扁平物体(吐司片、手机):竖起来沿厚度夹,需要换轴变体 reoriented_variant.py <src> x -> <类目>_thinz
        return (f"水平 {width * 100:.1f} cm 超宽,厚 {extents[2] * 100:.1f} cm:扁平物体,"
                "从边缘夹或竖放(竖放需换轴变体 _thinz)"), False
    if scalable:
        scale = max(SCALE_STEP, math.floor(MAX_WIDTH / width / SCALE_STEP) * SCALE_STEP)
        return f"可夹宽度 {width * 100:.1f} cm > 7 cm,入库时缩放到 s{round(scale * 100):03d}", scale < 0.6
    return f"可夹宽度 {width * 100:.1f} cm > 7 cm,官方资产不会自动缩放", True


# ---------------------------------------------------------------------------
# SimReady
# ---------------------------------------------------------------------------


def simready_search(query: str, k: int, url_regexp: str | None = None) -> list[dict]:
    body = {"limit": k, "return_usd_properties": True, "return_vision_generated_metadata": True}
    if url_regexp:
        body["filter_url_regexp"] = url_regexp
    else:
        body["description"] = query
    hits = json.loads(_http(SEARCH, body))["hits"]
    out = []
    for hit in hits:
        src = hit["source"]
        m = HF_PATH.fullmatch(src["path"])
        if not m:
            continue
        repo, rev, asset_dir, root_usd = m.groups()
        if repo not in SIMREADY_REPOS:
            continue
        props = {p["name"]: p["value"] for p in src.get("usd_properties", [])}
        vlm_items = src.get("plugin_thumbnail_to_vision_metadata_metadata_vlm_generated", [])
        vlm = {v["name"]: v.get("value_text") for v in vlm_items}
        extents = None
        if props.get("simready_metadata_asset_extents"):
            extents = [float(x) for x in re.findall(r"[-\d.eE]+", props["simready_metadata_asset_extents"])][:3]
        usd_dir, usd_name = root_usd.rsplit("/", 1)
        local_dir = f"simready-assets-{rev[:8]}"
        cmd = (f"hf download {repo} --repo-type dataset --revision {rev} --include '{asset_dir}/*' "
               f"--local-dir {local_dir}")
        out.append({
            "id": asset_dir,
            "source": "simready",
            "repo": repo,
            "cmd": cmd,
            "query": query,
            "score": round(hit.get("score", 0), 3),
            "extents": extents,
            "mass": _float(props.get("simready_metadata_mass")),
            "license": props.get("simready_metadata_asset_license"),
            "author": props.get("simready_metadata_author"),
            "rigid_bodies": props.get("simready_metadata_rigid_body_count"),
            "category": props.get("central:category"),
            "caption": (vlm.get("caption") or [""])[0],
            "thumb_url": f"{MIRROR}/datasets/{repo}/resolve/{rev}/{asset_dir}/{usd_dir}/.thumbs/256x256/{usd_name}.png",
        })
    return out


def _float(v):
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def fetch_thumb(group: str, cand: dict) -> None:
    url = cand.get("thumb_url")
    if not url:
        return
    # id 里的 ':' '/' 不能进文件名(页面里 "reuse:xx.png" 会被当成 URL scheme)
    path = group_dir(group) / "thumbs" / (re.sub(r"[^\w.-]", "_", cand["id"]) + ".png")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(_http(url, timeout=30))
        except (urllib.error.URLError, TimeoutError) as e:
            cand["thumb_error"] = str(e)
            return
    cand["thumb"] = f"thumbs/{path.name}"


# ---------------------------------------------------------------------------
# RoboDojo 官方 / 已选过的
# ---------------------------------------------------------------------------


def robodojo_revision(assets: dict) -> str | None:
    revs = Counter()
    for sources in assets.values():
        for cmd in sources.values():
            a = shlex.split(cmd) if cmd else []
            if a[2:3] == [ROBODOJO_REPO] and "--revision" in a:
                revs[a[a.index("--revision") + 1]] += 1
    return revs.most_common(1)[0][0] if revs else None


def _bbox(meta: dict):
    verts = (meta.get("geometry") or {}).get("aligned_bbox", {}).get("vertices")
    if not verts:
        return None
    return [round(max(v[i] for v in verts) - min(v[i] for v in verts), 4) for i in range(3)]


def robodojo_search(label: str, queries: list[str], k: int, rev: str | None) -> list[dict]:
    words = {w for q in [label, *queries] for w in re.split(r"[\s_\-]+", q.lower()) if len(w) > 2}
    scored = []
    for section in SECTIONS:
        sec = OBJECTS / section
        if not sec.is_dir():
            continue
        for cat in sorted(sec.iterdir()):
            if cat.name.startswith("SimReady_") or not cat.is_dir():
                continue
            cat_words = set(cat.name.lower().split("_"))
            for idx in sorted(cat.iterdir())[:k]:
                desc = ""
                if (idx / "description.json").exists():
                    desc = json.loads((idx / "description.json").read_text(encoding="utf-8")).get("description", "")
                    desc = " / ".join(desc) if isinstance(desc, list) else str(desc)
                # 类目名必须命中;描述只用来在同类目里排序(描述里的形容词太泛,单独命中噪声很大)
                cat_hit = len(words & cat_words)
                hit = cat_hit * 10 + len(words & set(re.split(r"\W+", desc.lower())))
                if cat_hit:
                    scored.append((hit, section, cat.name, idx.name, desc))
    out = []
    for hit, section, cat, idx, desc in sorted(scored, key=lambda x: -x[0])[:k]:
        path = f"{section}/{cat}/{idx}"
        cmd = (f"hf download {ROBODOJO_REPO} --repo-type dataset --revision {rev} "
               f"--include 'Assets/Object/RoboDojo/{path}/*' --local-dir robodojo-assets-{rev[:8]}") if rev else ""
        out.append({"id": path, "source": "robodojo", "repo": ROBODOJO_REPO, "cmd": cmd, "score": hit,
                    "license": "RoboDojo", "caption": desc, **local_facts(path)})
    return out


def local_facts(path: str) -> dict:
    """本地 RoboDojo 资产的尺寸 / 质量(``metadata.json``)。"""
    meta_path = OBJECTS / path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {"extents": _bbox(meta), "mass": (meta.get("physics") or {}).get("mass")}


def reuse_candidates(group: str, label: str, assets: dict) -> list[dict]:
    seen = {}
    for g, sources in assets.items():
        cmd = sources.get(label)
        if g != group and cmd:
            seen.setdefault(cmd, []).append(g)
    out = []
    for cmd, groups in seen.items():
        a = shlex.split(cmd)
        include = a[a.index("--include") + 1] if "--include" in a else "?"
        ref = include.removeprefix("Assets/Object/RoboDojo/").removesuffix("/*")
        repo = a[2] if len(a) > 2 else "?"
        cand = {"id": "reuse:" + ref, "source": "reuse", "repo": repo, "cmd": cmd}
        if repo == ROBODOJO_REPO:
            cand.update(local_facts(ref))
        elif repo in SIMREADY_REPOS:
            try:
                hit = simready_search("", 1, url_regexp=f".*/{re.escape(ref)}/simready_usd/.*")
            except (urllib.error.URLError, TimeoutError, KeyError):
                hit = []
            if hit:
                cand.update({k: v for k, v in hit[0].items() if k not in ("id", "source", "cmd", "query", "score")})
        cand["caption"] = f"已在 {', '.join(groups)} 中使用" + (f" · {cand['caption']}" if cand.get("caption") else "")
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------


def cmd_search(a) -> int:
    data = load_cands(a.group)
    assets = load_assets()
    entry = data["labels"].setdefault(a.label, {"queries": [], "grasped": False, "candidates": []})
    entry["grasped"] = entry["grasped"] or a.grasped
    entry["queries"] = list(dict.fromkeys([*entry["queries"], *a.query]))
    found = reuse_candidates(a.group, a.label, assets)
    found += robodojo_search(a.label, a.query, a.local_k, robodojo_revision(assets))
    for q in a.query:
        try:
            found += simready_search(q, a.k)
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            print(f"SimReady 搜索失败({q}): {e}", file=sys.stderr)
    have = {c["id"] for c in entry["candidates"]} | {c.get("cmd") for c in entry["candidates"] if c.get("cmd")}
    new = []
    for c in found:
        # 同一资产只列一次(reuse 排在前面,本地 / SimReady 里同一个命令跳过)
        if c["id"] in have or (c.get("cmd") and c["cmd"] in have):
            continue
        scalable = c["repo"] in SIMREADY_REPOS
        c["grasp_note"], c["grasp_warn"] = grasp_note(c.get("extents"), entry["grasped"], scalable)
        new.append(c)
        have |= {c["id"], c.get("cmd")}
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda c: fetch_thumb(a.group, c), new))
    entry["candidates"] += new
    entry["updated_at"] = now()
    save_cands(data)
    write_page(data)
    print_label(a.label, entry, assets.get(a.group, {}).get(a.label))
    return 0


def print_label(label: str, entry: dict, current: str | None) -> None:
    print(f"\n[{label}]{' (被抓取)' if entry.get('grasped') else ''}  查询: {entry.get('queries')}")
    for c in entry["candidates"]:
        ext = "×".join(f"{x * 100:.1f}" for x in c["extents"]) + " cm" if c.get("extents") else "尺寸未知"
        mark = "★" if current and c.get("cmd") == current else " "
        warn = " ⚠" if c.get("grasp_warn") else ""
        print(f"  {mark} {c['id']:<44} {c['source']:<8} {ext:<22} {c.get('caption', '')[:60]}"
              f"{'  ' + c['grasp_note'] if c.get('grasp_note') else ''}{warn}")


def cmd_show(a) -> int:
    data = load_cands(a.group)
    current = load_assets().get(a.group, {})
    for label, entry in data["labels"].items():
        if a.label and label != a.label:
            continue
        print_label(label, entry, current.get(label))
    return 0


def cmd_pick(a) -> int:
    data = load_cands(a.group)
    entry = data["labels"].get(a.label)
    if not entry:
        raise SystemExit(f"{a.label}: 还没有候选,先 search")
    cand = next((c for c in entry["candidates"] if c["id"] == a.candidate), None)
    if not cand:
        raise SystemExit(f"{a.label}: 没有候选 {a.candidate}(show 查看)")
    if not cand.get("cmd"):
        raise SystemExit(f"{a.candidate}: 没有下载命令(RoboDojo 官方 revision 未知)")
    assets = load_assets()
    # 同一 SimReady 类目在多个组里必须用同一个模型
    if cand["repo"] in SIMREADY_REPOS:
        clash = [g for g, s in assets.items() if g != a.group and s.get(a.label)
                 and shlex.split(s[a.label])[2] in SIMREADY_REPOS and s[a.label] != cand["cmd"]]
        if clash:
            raise SystemExit(f"{a.label}: {', '.join(clash)} 已用另一个 SimReady 模型;"
                             "要么选 reuse 候选,要么换 label 名")
    assets.setdefault(a.group, {})[a.label] = cand["cmd"]
    ASSETS_JSON.write_text(json.dumps(assets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    entry["picked"] = a.candidate
    save_cands(data)
    write_page(data)
    print(f"{a.group}.{a.label} ← {a.candidate}")
    return 0


def cmd_drop(a) -> int:
    data = load_cands(a.group)
    data["labels"].pop(a.label, None)
    save_cands(data)
    assets = load_assets()
    if assets.get(a.group, {}).pop(a.label, None) is not None:
        ASSETS_JSON.write_text(json.dumps(assets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_page(data)
    print(f"{a.group}.{a.label} 已移除")
    return 0


# ---------------------------------------------------------------------------
# 对照页
# ---------------------------------------------------------------------------


def write_page(data: dict) -> Path:
    current = load_assets().get(data["group"], {})
    sections = []
    for label, entry in data["labels"].items():
        cards = []
        for c in entry["candidates"]:
            picked = current.get(label) and c.get("cmd") == current.get(label)
            badge = " <b>已选</b>" if picked else ""
            img = (f'<img src="{html.escape(c["thumb"])}" alt="" loading="lazy">' if c.get("thumb")
                   else '<div class="noimg">无缩略图</div>')
            ext = " × ".join(f"{x * 100:.1f}" for x in c["extents"]) + " cm" if c.get("extents") else "尺寸未知"
            facts = [ext]
            if c.get("mass") is not None:
                facts.append(f"{c['mass']} kg")
            if c.get("rigid_bodies") not in (None, "1"):
                facts.append(f"{c['rigid_bodies']} 个刚体(可能带关节)")
            if c.get("license"):
                facts.append(html.escape(str(c["license"])))
            note = (f'<p class="{"warn" if c.get("grasp_warn") else "ok"}">{html.escape(c["grasp_note"])}</p>'
                    if c.get("grasp_note") else "")
            cards.append(f"""<div class="card{' picked' if picked else ''}">{img}
<div class="body"><code>{html.escape(c['id'])}</code> <span class="tag">{c['source']}</span>{badge}
<p>{html.escape(c.get('caption') or '')}</p><p class="muted">{' · '.join(facts)}</p>{note}
<p class="muted">{html.escape(c.get('category') or c['repo'])}</p></div></div>""")
        grasp = " · 被抓取" if entry.get("grasped") else ""
        sections.append(f"""<section><h2>{html.escape(label)}<small>{grasp} · 查询:
{html.escape(' / '.join(entry.get('queries', [])))}</small></h2><div class="grid">{''.join(cards)}</div></section>""")
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Asset Candidates</title><style>
:root{{--bg:#fafaf8;--fg:#1f2328;--muted:#6b7280;--line:#e5e7eb;--card:#fff;--pick:#059669;--warn:#b91c1c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181d;--fg:#e6e6e6;--muted:#9ca3af;--line:#2d3139;--card:#1e2128;
--pick:#34d399;--warn:#f87171}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:16px}}
h2{{margin:28px 0 8px}}h2 small{{font-size:13px;font-weight:normal;color:var(--muted);margin-left:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.card.picked{{border:2px solid var(--pick)}}.card img,.noimg{{width:100%;aspect-ratio:1;object-fit:contain;
background:#8881;display:flex;align-items:center;justify-content:center;color:var(--muted)}}
.body{{padding:8px 10px}}.body p{{margin:4px 0}}.muted{{color:var(--muted);font-size:12px}}
.tag{{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--muted)}}
.warn{{color:var(--warn)}}.ok{{color:var(--pick)}}code{{word-break:break-all}}
</style></head><body><h1>{html.escape(data['group'])} · 资产候选</h1>
<p class="muted">生成于 {now()} · 选定:
<code>asset_candidates.py pick {html.escape(data['group'])} &lt;label&gt; &lt;id&gt;</code></p>
{''.join(sections) or '<p>还没有候选</p>'}</body></html>
"""
    path = group_dir(data["group"]) / "candidates.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("search")
    sp.add_argument("group")
    sp.add_argument("label")
    sp.add_argument("--query", action="append", required=True, help="英文描述,可重复")
    sp.add_argument("--grasped", action="store_true", help="该物体会被抓取(检查可夹宽度)")
    sp.add_argument("--k", type=int, default=4, help="每个查询取 SimReady 前 k 个")
    sp.add_argument("--local_k", type=int, default=3, help="RoboDojo 官方最多取几个")
    sp = sub.add_parser("show")
    sp.add_argument("group")
    sp.add_argument("label", nargs="?")
    sp = sub.add_parser("pick")
    sp.add_argument("group")
    sp.add_argument("label")
    sp.add_argument("candidate")
    sp = sub.add_parser("drop")
    sp.add_argument("group")
    sp.add_argument("label")
    sp = sub.add_parser("page")
    sp.add_argument("group")
    a = p.parse_args()
    if a.cmd == "page":
        print(write_page(load_cands(a.group)))
        return 0
    return {"search": cmd_search, "show": cmd_show, "pick": cmd_pick, "drop": cmd_drop}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
