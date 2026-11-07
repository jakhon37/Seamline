"""Run the newest model on defect-free VisA boards. Every shape it proposes is a false alarm. Usage: clean_boards.py VISA/pcb1 PROJECT WORK OUT.json"""
import json, random, shutil, subprocess, sys
from pathlib import Path

visa, project, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
repo = Path(__file__).resolve().parents[2]
run = sorted((project / ".myseg/runs").iterdir())[-1]
(work / ".myseg/runs").mkdir(parents=True, exist_ok=True)
shutil.copy(project / ".myseg/project.json", work / ".myseg/project.json")
link = work / ".myseg/runs" / run.name
if not link.exists():
    link.symlink_to(run.resolve())
names = sorted(p.name for p in (visa / "Data/Images/Normal").iterdir())
random.seed(37)
chosen = random.sample(names, 30)
for n in chosen:
    shutil.copy(visa / "Data/Images/Normal" / n, work / n)
done = subprocess.run([str(repo / ".venv/bin/python"), str(repo / "engine/studio.py"), "predict", str(work), "--run", run.name],
                      input=json.dumps(chosen), capture_output=True, text=True, cwd=repo)
rows = [json.loads(l) for l in done.stdout.splitlines() if l.startswith("{")]
results = [r for r in rows if "name" in r]
shapes = [s for r in results for s in r.get("shapes", [])]
summary = {
    "clean_boards": len(results),
    "boards_with_a_false_alarm": sum(1 for r in results if r.get("shapes")),
    "false_alarm_shapes": len(shapes),
    "by_class": {c: sum(1 for s in shapes if s["label"] == c) for c in sorted({s["label"] for s in shapes})},
    "score_median": sorted(s["score"] for s in shapes)[len(shapes) // 2] if shapes else None,
    "errors": [r["error"] for r in results if "error" in r][:3],
    "per_board": {r["name"]: [(s["label"], round(s["score"], 2)) for s in r.get("shapes", [])] for r in results},
}
out.write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "per_board"}))
print(done.stderr[-300:] if done.returncode else "")
