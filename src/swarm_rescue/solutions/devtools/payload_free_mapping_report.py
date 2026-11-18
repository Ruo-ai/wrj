"""Render a payload-free mapping batch using only its saved observations and receipts."""

import argparse
import base64
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
from tempfile import mkdtemp
from zipfile import BadZipFile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer


SLOT_COLORS = ["#1a639d", "#d07716", "#009e73", "#bd5493", "#6b57ba"]
MAP_COLORS = ["#ffffff", "#9ba2a8", "#467cba", "#008e64", "#ddad36", "#c83931"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, decimals=0):
    return "—" if value is None else f"{value:,.{decimals}f}"


def display_path(path):
    return path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path


def read_round(batch, scheduled):
    folder = batch / scheduled["run_dir"]
    reasons = []
    receipts = {}
    try:
        hashes = read(folder / "integrity.json")
        for name in ("result.json", "resources.json", "validity.json", "trial.npz", "run.log"):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != hashes[name]:
                reasons.append(f"{name} 与运行收据哈希不符")
    except (OSError, ValueError, KeyError) as error:
        reasons.append(f"运行收据检查未完成（{error}）")
    for name in ("result", "validity", "resources"):
        try:
            receipts[name] = read(folder / f"{name}.json")
        except (OSError, ValueError) as error:
            receipts[name] = {}
            reasons.append(f"{name}.json 读取失败（{error}）")
    result = receipts["result"]
    if result and any(result.get(key) != scheduled[key] for key in ("map", "round", "seed")):
        reasons.append("结果身份与计划不符")
    if result and result.get("bombs") != 0:
        reasons.append("零载荷检查未通过")
    validity = receipts["validity"]
    if validity.get("valid") is not True:
        reasons.extend(str(reason) for reason in (validity.get("reasons") or ["缺少有效运行收据"]))
    return dict(plan=scheduled, folder=folder, result=result, resources=receipts["resources"],
                reasons=reasons, missed=None, last_active_step=None, empty_tail=None, data=None)


def load_maps(row):
    with np.load(row["folder"] / "trial.npz", allow_pickle=False) as archive:
        truth = archive["truth"] != 0
        pred = archive["pred"] != 0
        xy = archive["positions"]
        visible = archive["visible"] != 0 if "visible" in archive else None
    if truth.ndim != 2 or pred.shape != truth.shape or not truth.any():
        raise ValueError("墙体真值为空或栅格大小不一致")
    if visible is not None and visible.shape != truth.shape:
        raise ValueError("覆盖栅格大小不一致")
    if xy.ndim != 3 or xy.shape[1:] != (5, 2) or not len(xy):
        raise ValueError(f"轨迹形状异常 {xy.shape}")
    active = np.isfinite(xy).all(axis=2)
    if not active.any(axis=0).all():
        raise ValueError("存在整局无有效位置的槽位")
    if len(xy) != row["result"].get("steps"):
        raise ValueError("轨迹帧数与物理步数不符")
    row["last_active_step"] = int(np.flatnonzero(active.any(axis=1))[-1]) + 1
    row["empty_tail"] = len(xy) - row["last_active_step"]

    result = row["result"]
    scorer = MapExplorationScorer()
    scores = scorer.score(truth, pred, 0.0)
    row["missed"] = scores.wall.misses
    for name, expected in (("map_score", 100 * scores.s_map), ("wall_hits", scores.wall.hits),
                           ("charged_pixels", scores.charged_pixels)):
        if result.get(name) is None or not np.isclose(result[name], expected, atol=1e-6, rtol=0):
            row["reasons"].append(f"{name} 与保存的墙图重算值 {expected:g} 不符")
    surface, interior = scorer.split_regions(truth)
    distance = scorer.distance_field(truth)
    diff = np.zeros(truth.shape, dtype=np.uint8)
    diff[interior != 0] = 1
    diff[surface != 0] = 2
    diff[(surface != 0) & pred] = 3
    diff[pred & ~truth & (distance <= scorer.config.dead_zone)] = 4
    diff[pred & ~truth & (distance > scorer.config.dead_zone)] = 5
    return truth, diff, visible, xy


def paint_round(axes, row):
    if row["data"] is None:
        for ax in axes:
            ax.text(.5, .5, "本局缺少可绘制的数据", transform=ax.transAxes,
                    ha="center", va="center", color="#8c3735")
            ax.set_axis_off()
        return
    truth, diff, visible, xy = row["data"]
    h, w = truth.shape
    axes[0].imshow(diff, cmap=ListedColormap(MAP_COLORS), vmin=0, vmax=5, interpolation="nearest")
    if visible is not None:
        coverage = visible.astype("uint8")
        coverage[truth] = 2
        axes[1].imshow(coverage, cmap=ListedColormap(["white", "#96d5b5", "#434a50"]),
                       vmin=0, vmax=2, interpolation="nearest")
    else:
        axes[1].text(.5, .5, "覆盖轮廓未保存", transform=axes[1].transAxes, ha="center")
    axes[2].imshow(truth, cmap=ListedColormap(["white", "#61686f"]), vmin=0, vmax=1,
                   interpolation="nearest")
    for slot, color in enumerate(SLOT_COLORS):
        track = xy[:, slot]
        axes[2].plot(track[:, 0] + w / 2, h / 2 - track[:, 1], color=color, linewidth=1.05)
    for ax in axes:
        ax.set(xlim=(-.5, w-.5), ylim=(h-.5, -.5), aspect="equal", xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_color("#92999e")


def draw_map(rows, path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    fig, matrix = plt.subplots(len(rows), 3, figsize=(15.8, 4.3 * len(rows) + 1.1), squeeze=False)
    name = rows[0]["plan"]["map"]
    versions = sorted({r["result"]["controller"] for r in rows if r["result"].get("controller")})
    fig.suptitle(f"{name}   |   {' / '.join(versions)}   |   无载荷建图", fontsize=17, weight="bold")
    for row, axes in zip(rows, matrix):
        r, item = row["result"], row["plan"]
        paint_round(axes, row)
        tag = f"r{item['round']} · seed {item['seed']} · " + ("异常" if row["reasons"] else "有效")
        texts = (f"墙体估计差异   M {number(r.get('map_score'), 2)}",
                 f"轨迹邻域覆盖   {number(r.get('coverage_percent'), 2)}%",
                 f"五机实际轨迹   落点数 {number(r.get('bombs'))}")
        for ax, subtitle in zip(axes, texts):
            ax.set_title(f"{tag}\n{subtitle}", fontsize=10, pad=7, linespacing=1.5)
    patches = [Patch(color=c, label=t) for c, t in zip(MAP_COLORS[1:],
               ("墙体内部", "漏画表面", "命中表面", "免罚预测", "受罚误画"))]
    patches.append(Patch(color="#96d5b5", label="轨迹邻域覆盖"))
    tracks = [Line2D([], [], color=c, lw=2, label=f"槽位 {i}") for i,c in enumerate(SLOT_COLORS)]
    fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(.5, .053), ncol=6, frameon=False)
    fig.legend(handles=tracks, loc="lower center", bbox_to_anchor=(.5, .016), ncol=5, frameon=False)
    fig.subplots_adjust(left=.025, right=.975, bottom=.12, top=.86, hspace=.32, wspace=.085)
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def write_replay(output, row, truth, xy):
    """Write a self-contained, scrubbable trace viewer with the original float32 frames."""
    item = row["plan"]
    h, w = truth.shape
    wall_png = io.BytesIO()
    Image.fromarray(np.where(truth, 72, 255).astype(np.uint8)).save(wall_png, "PNG")
    data = dict(width=w, height=h, frames=len(xy), slots=xy.shape[1], colors=SLOT_COLORS,
                title=f"{item['map']} · r{item['round']} · seed {item['seed']} · "
                      f"{row['result']['controller']} · {'异常' if row['reasons'] else '有效'}",
                bombs=row["result"].get("bombs"), wall=base64.b64encode(wall_png.getvalue()).decode(),
                positions=base64.b64encode(np.ascontiguousarray(xy, dtype="<f4").tobytes()).decode())
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>无载荷建图轨迹回放</title><style>
body{max-width:1100px;margin:1rem auto;padding:0 1rem;font:1rem system-ui;color:#27323d}
h1{font-size:1.2rem}canvas{display:block;max-width:100%;max-height:72vh;border:1px solid #bbb}
nav{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin:1rem 0}input{flex:1;min-width:200px}
</style><h1 id="title"></h1><p>原生仿真保存的逐步轨迹。灰色为墙体，各颜色为无人机槽位。</p>
<canvas id="canvas"></canvas><nav><button id="play">播放</button><label>步 <output id="step"></output></label>
<input id="seek" type="range" min="0" value="0" aria-label="选择物理步"><label>速度
<select id="speed"><option value="1">30 步/秒</option><option value="4">120 步/秒</option>
<option value="16">480 步/秒</option></select></label></nav><p id="legend"></p>
<script>const data=__DATA__, el=id=>document.getElementById(id);
el("title").textContent=data.title;
const ctx=el("canvas").getContext("2d"), wall=new Image(), seek=el("seek"), button=el("play");
ctx.canvas.width=data.width; ctx.canvas.height=data.height; seek.max=data.frames-1;
const raw=Uint8Array.from(atob(data.positions),c=>c.charCodeAt(0)), xy=new DataView(raw.buffer);
for(const [i,color] of data.colors.entries()){const item=document.createElement("span");
item.textContent="● 槽位 "+i+"　";item.style.color=color;el("legend").append(item)}
el("legend").append("　实际落点数 "+(data.bombs ?? "未记录"));
function point(t,s){const idx=(t*data.slots+s)*8;return [
xy.getFloat32(idx,true)+data.width/2,data.height/2-xy.getFloat32(idx+4,true)]}
function draw(){const t=Number(seek.value);ctx.drawImage(wall,0,0);ctx.lineWidth=2;
for(let s=0;s<data.slots;s++){ctx.strokeStyle=ctx.fillStyle=data.colors[s];ctx.beginPath();let active=false;
for(let i=0;i<=t;i++){const [x,y]=point(i,s);if(!Number.isFinite(x+y)){active=false;continue}
if(active){ctx.lineTo(x,y)}else{ctx.moveTo(x,y)}active=true}ctx.stroke();
const [x,y]=point(t,s);if(Number.isFinite(x+y)){ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fill()}}
el("step").textContent=(t+1)+" / "+data.frames}
let playing=false,previous=0,fraction=0,animation=0;
function stop(){playing=false;cancelAnimationFrame(animation);button.textContent="播放"}
function tick(now){if(!playing)return;fraction+=(now-previous)*0.03*Number(el("speed").value);previous=now;
const jump=Math.floor(fraction);if(jump){fraction-=jump;seek.value=Math.min(Number(seek.max),Number(seek.value)+jump);draw()}
if(seek.value===seek.max){stop()}else{animation=requestAnimationFrame(tick)}}
button.disabled=seek.disabled=true;button.onclick=()=>{if(playing){stop();return}if(seek.value===seek.max)seek.value=0;
playing=true;previous=performance.now();fraction=0;button.textContent="暂停";draw();animation=requestAnimationFrame(tick)};
seek.oninput=()=>{stop();draw()};wall.onload=()=>{draw();button.disabled=seek.disabled=false};
wall.src="data:image/png;base64,"+data.wall;</script></html>"""
    payload = json.dumps(data, ensure_ascii=True).replace("<", r"\u003c")
    output.write_text(html.replace("__DATA__", payload), encoding="utf-8")


def replay(batch, run_name):
    batch = batch.resolve()
    schedule = read(batch / "plan.json")["schedule"]
    item = next((p for p in schedule if p["run_dir"] == run_name), None)
    if item is None:
        raise ValueError(f"Unknown scheduled run {run_name}")
    row = read_round(batch, item)
    truth, _, _, xy = load_maps(row)
    output = Path(mkdtemp(prefix="track_replay_", dir=batch)) / f"{run_name}.html"
    write_replay(output, row, truth, xy)
    print(display_path(output))
    return 0


def make_report(batch):
    batch = batch.resolve()
    plan_path = batch / "plan.json"
    plan = read(plan_path)
    schedule = plan["schedule"]
    if not schedule:
        raise ValueError("计划中没有测试局")
    output = Path(mkdtemp(prefix="mapping_review_", dir=batch))
    groups = defaultdict(list)
    for item in schedule:
        groups[item["map"]].append(read_round(batch, item))
    stage = "短程冒烟" if plan["smoke"] else "完整测试"
    lines = [f"# 无载荷建图{stage}", "", f"批次 `{batch.name}`，共计划 {len(schedule)} 局。"
             "独立控制器的各机估计在离线端合成。M 按保存的估计墙图与真值重算。"
             "覆盖为引擎根据实际轨迹计算的邻域覆盖率。本次未评测红蓝对抗分。", "",
             f"计划 SHA256 为 `{hashlib.sha256(plan_path.read_bytes()).hexdigest()}`。[冻结计划](../plan.json)", "",
             f"每局串行运行，分配 {plan['cpu_count']} 个 CPU、{plan['memory_bytes']:,} 字节内存，"
             f"外层进程上限 {plan['outer_timeout_seconds']} 秒。"
             "各轮均启用无 GPS 和无通信区。覆盖按引擎 200 px 轨迹邻域口径记录。"
             "缺定位占比的分母为存活机实际执行的控制回调数，有雷达但缺 GPS 或罗盘时计为缺定位。"
             "每帧记录物理步结束时仍在场的槽位。最后一次有位置的帧为末次有机步，其后的连续帧为空场尾段。"
             "本批没有单独计数墙体和机间物理接触。", ""]
    all_rows = []
    for name, rows in groups.items():
        rows.sort(key=lambda r: r["plan"]["round"])
        for row in rows:
            try:
                row["data"] = load_maps(row)
            except (OSError, KeyError, ValueError, TypeError, BadZipFile) as error:
                row["reasons"].append(f"图件数据检查失败（{error}）")
        figure = f"{name}_mapping.png"
        draw_map(rows, output / figure)
        lines += [f"## {name}", "", "| 轮 / 种子 | 有效 | M | 命中 / 漏画 / 受罚像素 | 覆盖 | 步数 | "
                  "运行秒 / 进程秒 | 进程组峰值 MB | 落点 | 停止原因 |",
                  "|---|---|---:|---|---:|---:|---|---:|---:|---|"]
        for row in rows:
            r, p, res = row["result"], row["plan"], row["resources"]
            peak = res.get("peak_job_commit_bytes")
            lines.append(f"| r{p['round']} / {p['seed']} | {'否' if row['reasons'] else '是'} | "
                         f"{number(r.get('map_score'), 2)} | {number(r.get('wall_hits'))} / "
                         f"{number(row['missed'])} / {number(r.get('charged_pixels'))} | "
                         f"{number(r.get('coverage_percent'), 2)}% | {number(r.get('steps'))} | "
                         f"{number(r.get('wall_seconds'), 2)} / {number(res.get('process_seconds'), 2)} | "
                         f"{number(peak / 1e6 if peak is not None else None, 2)} | {number(r.get('bombs'))} | "
                         f"{r.get('stop_reason', '—')} |")
        lines.extend(["", "| 轮 | 最终健康，槽位 0–4 | 移除槽数 | 末次有机步 | 空场尾段（步） | 缺定位 / 实际回调 | 缺定位占比 | 缺雷达 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            r = row["result"]
            stats = r.get("mapping_stats", [])
            calls = sum(s["calls"] for s in stats)
            missing = sum(s["missing_pose_scans"] for s in stats)
            no_lidar = sum(s["missing_scan"] for s in stats) if stats else None
            removed = r.get("removed_slots")
            health = " / ".join(str(h) for h in r.get("final_health", [])) or "—"
            fraction = f"{missing / calls:.2%}" if calls else "—"
            denominator = f"{missing:,} / {calls:,}" if stats else "—"
            lines.append(f"| r{row['plan']['round']} | {health} | "
                         f"{number(len(removed) if removed is not None else None)} | "
                         f"{number(row['last_active_step'])} | {number(row['empty_tail'])} | "
                         f"{denominator} | {fraction} | {number(no_lidar)} |")
        lines.extend(["", f"![{name} 无载荷各轮建图、覆盖、五机轨迹与落点数]({figure})", ""])
        for row in rows:
            link = "../" + Path(row["plan"]["run_dir"]).as_posix()
            note = "。".join(row["reasons"]) if row["reasons"] else "检查通过"
            replay_link = ""
            if row["data"] is not None and row["result"].get("controller"):
                viewer = f"{row['plan']['run_dir']}.html"
                write_replay(output / viewer, row, row["data"][0], row["data"][3])
                replay_link = f" · [直接播放本局轨迹]({viewer})"
            if row["data"] is not None and row["data"][2] is None:
                note += "。覆盖仅保存数值，缺少空间分布"
            lines.append(f"- r{row['plan']['round']} {note}。[原始产物]({link}/){replay_link}")
            del row["data"]
        lines.append("")
        all_rows.extend(rows)
    passed = sum(not row["reasons"] for row in all_rows)
    lines.insert(4, f"本次复核有效 {passed}/{len(schedule)} 局，异常 {len(schedule)-passed} 局。\n")
    lines.extend(["## 从保存的数据重新生成", "", "在项目根目录执行下列 PowerShell 命令。"
                  "每次生成独立的新图册目录，仿真数据来自各局的 trial.npz。", "",
                  "```powershell",
                  f".\\.venv\\Scripts\\python.exe -B -m swarm_rescue.solutions.devtools.payload_free_mapping_report '{batch}'",
                  "```", "", "在项目根目录执行下列命令可直接打开本次已生成的逐局 HTML。"
                  "回放支持播放、暂停、变速与定位到指定物理步，使用观测器记录的墙体和每槽逐帧坐标。", "",
                  "```powershell"])
    for row in all_rows:
        viewer = output / f"{row['plan']['run_dir']}.html"
        if viewer.is_file():
            lines.append(f"Invoke-Item -LiteralPath '{display_path(viewer)}'")
    lines.extend(["```", ""])
    report = output / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"{display_path(report)} | valid={passed}/{len(schedule)}")
    return 0 if passed == len(schedule) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--replay", metavar="RUN_DIR", help="Write an offline HTML trace replay for one round")
    args = parser.parse_args()
    raise SystemExit(replay(args.batch, args.replay) if args.replay else make_report(args.batch))
