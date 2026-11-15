"""Regression test for the gate evaluator. Run from anywhere:
    python -I tools/smoke_evaluate.py
Checks (1) the python embedded in gate0_1.sh compiles, (2) probe.py compiles,
(3) evaluate() produces CORRECT rows and projection math on a synthetic battery
result (W2 #12: real asserts, not a rubber stamp). All output goes to a temp dir —
never into recon/gates/ (W2 #12: the old glob-rm could destroy real gate reports).
"""
import importlib.util, sys, argparse, py_compile, pathlib, tempfile, shutil, json

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 1. syntax-check the python embedded in the bash battery
src = (ROOT / "tools" / "ps5probe" / "gate0_1.sh").read_text(encoding="utf-8")
py = src.split("<<'PYEOF'", 1)[1].split("PYEOF", 1)[0].lstrip("\n")
p = ROOT / "tools" / "ps5probe" / "_n7extract.py"
p.write_text(py, encoding="utf-8")
py_compile.compile(str(p), doraise=True)
p.unlink()
print("embedded N7 python: OK")

# 2. probe.py itself compiles
py_compile.compile(str(ROOT / "tools" / "ps5probe" / "probe.py"), doraise=True)
print("probe.py compile: OK")

# 3. smoke-test the evaluator with a synthetic battery result (incl. N6/N7)
spec = importlib.util.spec_from_file_location("p", ROOT / "tools" / "ps5probe" / "probe.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tmp = tempfile.mkdtemp(prefix="ps5-smoke-")
m.REPORT_DIR = pathlib.Path(tmp)              # keep evaluate() out of the real recon/gates/ dir
# S8: real gate reports live in recon/gates/ (committed evidence) — assert smoke adds none
_pre_reports = set((ROOT / "recon" / "gates").glob("ps5-gate-*"))
js = {"os": "Linux 6.14.0-ps5 x86_64", "kernel_cmdline": "amdgpu.gttsize=14750",
      "pci_vga": "00:00.0 VGA [1002:1503]", "device_id": "0x7340",
      "vk_summary": "GPU0: AMD Oberon (RADV GFX1013)", "vk_ext_host": 1,
      "ttm_pages_limit": 4194304, "ttm_page_pool_size": 4194304,
      "amdgpu_gttsize": "14750", "vram_total": "16106127360",
      "block_devs": "nvme0n1:161061273600:SSD 825 |nvme1n1:2000398933504:WD_BLACK SN850 |",
      "fio_size_mb": 102400, "gtt_total": "15032385536",
      "g1_rand_scratch_mb": 102400,
      "g1_target": "auto-largest-free", "g1_device": "/dev/nvme0", "g1_device_model": "SSD 825",
      "scratch_mount": "/data", "scaling_governor": "performance", "cpu_boost": "1",
      "fio_seq1m_mb_s": 5320, "fio_rand3m_mb_s": 4010, "fio_rand4k_mb_s": 310,
      "buffered_reread_mb_s": 16900,
      "n6_devices": "nvme0n1+nvme1n1", "n6_a_mb_s": 5110, "n6_b_mb_s": 6240, "n6_aggregate_mb_s": 11350,
      "n7_hit_pct": 84.2, "n7_tokens": 356, "n7_store_gb": 40.0,
      "n7_miss_gb": 18.9, "n7_ssd_mb_s": 2158.0, "n7_slots": 13653, "n7_slots_per_layer": 324, "n7_cache_mb": 3072,
      "n7_mem_available_mb": 11264, "n7_cgroup_current": 3221225472, "n7_chunk_bytes": 2764800,
      "n7_mode": "cgroup memory.high=3072M", "n7_zipf_s": "1.1", "n7_ran": "1"}
a = argparse.Namespace(host="10.0.0.201", expert_gb=71.7, k=10, experts=512, gib_per_token=1.24)
m.evaluate(js, a)
w = json.loads((pathlib.Path(tmp) / sorted(pathlib.Path(tmp).glob("ps5-gate-*.json"))[0].name).read_text())
rows = w["rows"]; proj = w["projection"]

def rows_named(sub):
    return [r for r in rows if sub in r[1]]

# --- structural checks ---
for g in ("G0", "G1", "N6", "N7"):
    assert any(r[0] == g for r in rows), f"no {g} rows emitted"
assert rows_named("GPU identity")[0][2] is True, "GPU identity should PASS on GFX1013 fixture"
assert rows_named("NVMe store device present")[0][2] is True, "an NVMe store device should PASS G0"
assert "N6 stripe possible" in rows_named("NVMe store device present")[0][3], "two NVMe -> stripe note"
assert "gtt_total=14.00 GiB" in rows_named("ttm pages_limit")[0][3], "GTT must come from sysfs gtt_total (gttsize param is root-only)"
# S10: no canned verdict strings — every verdict is ledger-computed for Q4 and EXL3 1.66
_all = json.dumps(rows)
for stale in ("PIVOT", "35B-A3B", "out of reach", "band tops ~30", "REAP-pack band"):
    assert stale not in _all, f"stale canned verdict string survived: {stale}"
assert abs(m.need_hit(4840, 0.443) - 0.746) < 1e-3, m.need_hit(4840, 0.443)

# --- G1 must attribute the device (W2 #17 fix) ---
seq_row = rows_named("sequential read")[0]
assert seq_row[2] is True and "5320" in seq_row[3], seq_row
assert "nvme0" in seq_row[3] and "SSD 825" in seq_row[3], "G1 row must surface which device it measured"
assert rows_named("boost state")[0][2] is True, "clock-state row must be present (W2 #15 fix)"

# --- projection math with the W4 constant (G=1.24), no hidden MTP multiplier ---
# S10 unit fix: fio MB/s are decimal -> GiB/s via /1073.741824 (the old /1000
# inflated every projection ~7.4%).
MB_GIB = 1073.741824
assert proj["cold(h=0)"] == round(5320 / MB_GIB / 1.24, 2), proj
key = "streamed(h=0.84 measured)"
assert proj[key] == round(11350 / MB_GIB / (1.24 * (1 - 0.842)), 1), proj   # stripe dominates (N6 passed)
assert "with MTP x1.44 (N4 UNMEASURED)" in proj, "MTP multiplier must be labelled unmeasured"

# --- N7 is a MEASUREMENT row (None), not a pass/fail gate ---
n7 = rows_named("expert-cache replay")[0]
assert n7[2] is None, "N7 must not print PASS/FAIL (re-scoped to a curve measurement)"
assert "84.2" in n7[1], "N7 row name must carry the measured hit %"

# --- key-read canaries (W9-06/-07: evaluator must read the keys the gate EMITS) ---
assert "boost=1" in rows_named("boost state")[0][3], "boost row must read cpu_boost, not boost_state (W9-06)"
assert "324 slots/layer" in n7[1], "N7 row must render n7_slots_per_layer under the slots/label (W9-07)"
assert "MemAvailable 11264 MiB" in n7[3], "N7 row must surface n7_mem_available_mb in MiB (S10: mem from artifact, not docs)"
assert "pool reached 3.00 GiB of cap 3.00 GiB" in n7[3], "N7 row must show the reached pool (hit tracks reached, not cap)"
assert "EXL3-1.66" in n7[3] and "40 needs h>=" in n7[3], "N7 row must price both packs from the ledger"
assert "under-credits" in n7[3], "N7 row must flag Q4-size slot geometry"
assert any(k.startswith("EXL3-1.66 streamed") for k in proj), proj
assert rows_named("engine-shaped randread")[0][2] is True, "store-scale span keeps the engine-shaped pass/fail"
assert "rand scratch 102400 MiB" in rows_named("engine-shaped randread")[0][3], "S11: rand row must cite its scratch (SLC-honest provenance)"

# --- SPAN-LIMITED fio (S10: 1 GiB span reads like seq; never quote it as engine-shaped) ---
tmp4 = tempfile.mkdtemp(prefix="ps5-smoke4-")
m.REPORT_DIR = pathlib.Path(tmp4)
js4 = dict(js); js4["fio_size_mb"] = 1024; js4["g1_rand_scratch_mb"] = 0  # pre-S11 run: no rand scratch kv
m.evaluate(js4, a)
w4 = json.loads((pathlib.Path(tmp4) / sorted(pathlib.Path(tmp4).glob("ps5-gate-*.json"))[0].name).read_text())
lim = [r for r in w4["rows"] if "SPAN-LIMITED" in r[1]]
assert lim and lim[0][2] is None and "NOT engine-shaped" in lim[0][3], w4["rows"]
assert "rand " not in [r for r in w4["rows"] if r[0] == "N7"][0][3], "span-limited rand must not price the N7 band"
shutil.rmtree(tmp4, ignore_errors=True)

# --- S11: the rand row's span is ITS OWN scratch, not the seq scratch ---
tmp5 = tempfile.mkdtemp(prefix="ps5-smoke5-")
m.REPORT_DIR = pathlib.Path(tmp5)
js5 = dict(js); js5["fio_size_mb"] = 1024  # seq/4k scratch tiny; rand scratch still 100G
m.evaluate(js5, a)
w5 = json.loads((pathlib.Path(tmp5) / sorted(pathlib.Path(tmp5).glob("ps5-gate-*.json"))[0].name).read_text())
rr = rows_named("engine-shaped randread")
assert rr and "over 102400 MiB span" in rr[0][1], w5["rows"]
shutil.rmtree(tmp5, ignore_errors=True)

assert (pathlib.Path(tmp) / ("ps5-gate-" + w["ts"] + ".md")).exists()

# --- N7-LESS battery (W9-12): report must survive a skipped/failed N7 (W9-02 regression) ---
tmp2 = tempfile.mkdtemp(prefix="ps5-smoke2-")
m.REPORT_DIR = pathlib.Path(tmp2)
js2 = {k: v for k, v in js.items() if not k.startswith("n7_")}
m.evaluate(js2, a)  # used to raise TypeError ('%.2f' % None) and lose the whole report
w2 = json.loads((pathlib.Path(tmp2) / sorted(pathlib.Path(tmp2).glob("ps5-gate-*.json"))[0].name).read_text())
proj2 = w2["projection"]
assert "streamed(h=0.77 expected)" in proj2 and "streamed(h=0.88 expected)" in proj2, proj2
assert "with MTP x1.44 (N4 UNMEASURED)" in proj2, "MTP line must still print (labelled) without N7"
assert not any(r[0] == "N7" for r in w2["rows"]), "no N7 row when the battery did not report one"

# --- colon-less block_devs (W9-08: no IndexError from one odd sysfs read) ---
js3 = dict(js2); js3["block_devs"] = "sda|sdb"
m.evaluate(js3, a)  # must not raise

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(tmp2, ignore_errors=True)
assert set((ROOT / "recon" / "gates").glob("ps5-gate-*")) == _pre_reports, "smoke must not leave (or clobber) real reports"
print("SMOKE OK")
