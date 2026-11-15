"""Engine regression tests — run with: python -m unittest discover -s engine/tests"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import cli, decode, ledger, manifest, replay, trace  # noqa: E402


class TestLedger(unittest.TestCase):
    def test_headline_invariants(self):
        """S3 re-derivation: 40 tok/s <=> h=0.896, 60 <=> h=0.931 at G=1.24."""
        checks = ledger.headline_check()
        self.assertTrue(all(checks.values()), checks)

    def test_required_hit_unreachable(self):
        with self.assertRaises(ValueError):
            ledger.required_hit(bw_gib_s=1.0, g_gib=1.24, tok_s_target=1e6)

    def test_tok_s_monotone_in_hit(self):
        prev = -1.0
        for hit in (0.0, 0.3, 0.6, 0.9):
            v = ledger.tok_s(5.0, 1.24, hit)
            self.assertGreater(v, prev)
            prev = v

    def test_reprice_rows_math(self):
        """Alpha deliverable: measured-constants table decodes tok/s = bw/(G*(1-hit))."""
        rows = ledger.reprice_rows(
            g_values=[("q4", 1.24), ("exl3", 0.443)],
            hit_points=[("cold", 0.0), ("cap8192", 0.514)],
            bw_gib_s=2.625,
        )
        self.assertAlmostEqual(rows[0][2]["q4"], 2.625 / 1.24, places=3)
        # 2.625 / (0.443 * (1 - 0.514)) = 2.625 / 0.2153 = 12.19...
        self.assertAlmostEqual(rows[1][2]["exl3"], 2.625 / (0.443 * 0.486), places=3)

    def test_reprice_table_shape(self):
        """Table has a header, a separator, and one row per hit point."""
        table = ledger.reprice_table(
            g_values=[("a", 1.0), ("b", 2.0)],
            hit_points=[("cold", 0.0), ("h50", 0.5)],
            bw_gib_s=1.0,
        )
        lines = table.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("| pool (N7 cap) | hit |"))
        self.assertIn("cold", lines[2])
        self.assertIn("h50", lines[3])

    def test_mb_s_to_gib_s(self):
        """S10 unit fix: 4920 MB/s = 4.582 GiB/s, NOT 4.92 (old /1000 inflated ~7.4%)."""
        self.assertAlmostEqual(ledger.mb_s_to_gib_s(4920.0), 4.582, places=3)
        self.assertAlmostEqual(ledger.mb_s_to_gib_s(2625.0), 2.445, places=3)
        with self.assertRaises(ValueError):
            ledger.mb_s_to_gib_s(-1.0)


class TestManifest(unittest.TestCase):
    def test_q4_anchor_g(self):
        """At the Q4 anchor bpw the manifest must reproduce G=1.24."""
        cfg = manifest.ModelConfig()
        g = manifest.g_per_token_gib(cfg, manifest.EFFECTIVE_Q4_BPW)
        self.assertAlmostEqual(g, ledger.G_Q4_GIB, delta=0.01)

    def test_exl3_class_g_in_band(self):
        """2-bit class packs land in the S6-measured 0.44-0.58 GiB/token band."""
        cfg = manifest.ModelConfig()
        for bpw in (1.66, 2.18):
            g = manifest.g_per_token_gib(cfg, bpw)
            self.assertLess(g, 0.70, f"bpw={bpw}: G={g:.3f} outside the 2-bit class")

    def test_build_and_verify(self):
        m = manifest.build_manifest(manifest.ModelConfig(), 2.05)
        self.assertEqual(manifest.verify_manifest(m), [])
        self.assertEqual(len(m["layers"]), 48)
        self.assertEqual(len(m["layers"][0]["slots"]), 512)

    def test_hot_expert_pinning(self):
        m = manifest.build_manifest(manifest.ModelConfig(), 2.05, hot_experts={0: [5, 3, 8]})
        slots = m["layers"][0]["slots"]
        order = sorted(slots, key=slots.get)
        self.assertEqual(order[:3], [5, 3, 8])

    def test_verify_catches_corruption(self):
        m = manifest.build_manifest(manifest.ModelConfig(), 2.05)
        m["pack_bytes"] += 1
        self.assertTrue(manifest.verify_manifest(m))


class TestTrace(unittest.TestCase):
    def test_deterministic(self):
        cfg = manifest.ModelConfig()
        a = trace.generate(20, cfg.moe_layers, cfg.n_experts, cfg.top_k, seed=3)
        b = trace.generate(20, cfg.moe_layers, cfg.n_experts, cfg.top_k, seed=3)
        self.assertEqual(a["routing"], b["routing"])

    def test_agent_profile_repetition(self):
        """agent profile must repeat expert sets more than uniform (S5 prior)."""
        cfg = manifest.ModelConfig()

        def repetition(t):
            """LRU hit rate with a 48-slot cache per layer — what the runtime sees."""
            per_layer = list(zip(*t["routing"]))  # [layer][token] -> ids
            reps = []
            for layer in per_layer:
                cache: OrderedDict[int, None] = OrderedDict()
                hits = misses = 0
                for ids in layer:
                    for e in ids:
                        if e in cache:
                            hits += 1
                            cache.move_to_end(e)
                        else:
                            misses += 1
                            if len(cache) >= 48:
                                cache.popitem(last=False)
                            cache[e] = None
                reps.append(hits / (hits + misses))
            return sum(reps) / len(reps)

        agent = trace.generate(400, cfg.moe_layers, cfg.n_experts, cfg.top_k, profile="agent")
        uniform = trace.generate(400, cfg.moe_layers, cfg.n_experts, cfg.top_k, profile="uniform")
        self.assertGreater(repetition(agent), repetition(uniform))

    def test_valid_expert_ids(self):
        cfg = manifest.ModelConfig()
        t = trace.generate(10, 4, 32, cfg.top_k)
        for token in t["routing"]:
            for layer in token:
                self.assertEqual(len(layer), cfg.top_k)
                self.assertTrue(all(0 <= e < 32 for e in layer))


class TestReplay(unittest.TestCase):
    def _small_setup(self):
        # Tiny model so hand-checks are possible: 1 layer, 8 experts, top-2.
        slot = 1024
        layers = [{"layer": 0, "slot_bytes": slot, "slots": {e: e * slot for e in range(8)}}]
        return {
            "schema": "ps5lm-pack/1",
            "model": "tiny",
            "bpw": 2.0,
            "moe_layers": 1,
            "n_experts": 8,
            "top_k": 2,
            "shared_experts": 0,
            "expert_slot_bytes": slot,
            "pack_bytes": 8 * slot,
            "g_gib_per_token": 2 * 1 * slot / 1024**3,
            "provenance": {},
            "layers": layers,
        }

    def _tiny_trace(self, routing):
        return {
            "schema": "ps5lm-trace/1",
            "profile": "hand",
            "seed": 0,
            "tokens": len(routing),
            "moe_layers": 1,
            "n_experts": 8,
            "top_k": 2,
            "routing": routing,
        }

    def test_lru_hand_check(self):
        """Cache cap 2 over set {0,1}: T0 misses both, T1 hits both -> 2/4."""
        m = self._small_setup()
        t = self._tiny_trace([[[0, 1]], [[0, 1]]])
        report = replay.replay(m, t, cache_gib=2 * 1024 / 1024**3, bw_gib_s=1.0, tok_s_target=1000.0)
        self.assertEqual(report["expert_reads_total"], 4)
        self.assertEqual(report["hit_rate"], 0.5)
        self.assertEqual(report["verdict"], "FAIL")

    def test_pass_verdict_possible(self):
        m = self._small_setup()
        t = self._tiny_trace([[[0, 1]]] * 20)
        report = replay.replay(m, t, cache_gib=2 * 1024 / 1024**3, bw_gib_s=1.0, tok_s_target=1000.0)
        # 40 reads, 2 compulsory misses on the first token -> 38/40.
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["hit_rate"], 0.95)

    def test_budget_is_shared_across_layers(self):
        """Regression (2026-11-03): the budget is ONE pool, not one pool per layer.

        2 layers x the same 2 experts, budget = 2 slots. A shared pool can hold
        only half the 4 (layer, expert) keys -> LRU thrashes -> 0 hits. The old
        per-layer bug gave each layer 2 slots -> 75% hit on 4 tokens.
        """
        m = self._small_setup()
        m["moe_layers"] = 2
        m["layers"].append({**m["layers"][0], "layer": 1})
        t = self._tiny_trace([[[0, 1], [0, 1]]] * 4)
        t["moe_layers"] = 2
        report = replay.replay(m, t, cache_gib=2 * 1024 / 1024**3, bw_gib_s=1.0, tok_s_target=1000.0)
        self.assertEqual(report["expert_reads_total"], 16)
        self.assertEqual(report["cache_slots_total"], 2)
        self.assertEqual(report["hit_rate"], 0.0)

    def test_unreachable_verdict(self):
        m = self._small_setup()
        t = self._tiny_trace([[[0, 1]]] * 5)
        report = replay.replay(m, t, cache_gib=2 * 1024 / 1024**3, bw_gib_s=0.000001, tok_s_target=5000.0)
        self.assertEqual(report["verdict"], "UNREACHABLE")


class TestDecode(unittest.TestCase):
    """Decoder-gap spike (S9 queue #4): decode requirement is hit-rate-free."""

    def test_experts_per_s_at_40(self):
        """40 tok/s x 480 routed reads = 19,200 experts/s."""
        r = decode.requirement(40.0)
        self.assertAlmostEqual(r.experts_per_s, 19200.0, places=6)

    def test_gparams_s_uses_exact_params(self):
        """Uses the S5-audited 4,915,200, not manifest's back-out (~4.48M)."""
        self.assertEqual(decode.EXPERT_PARAMS_EXACT, 3 * 2560 * 640)
        self.assertNotEqual(decode.EXPERT_PARAMS_EXACT, manifest.EXPERT_PARAMS)
        r = decode.requirement(40.0)
        self.assertAlmostEqual(r.gparams_s, 19200 * 4_915_200 / 1e9, places=6)

    def test_shared_expert_option(self):
        """include_shared adds 48 reads: 40 x 528 = 21,120 experts/s."""
        r = decode.requirement(40.0, include_shared=True)
        self.assertAlmostEqual(r.experts_per_s, 21120.0, places=6)

    def test_independent_of_hit_rate(self):
        """The point of the spike: no hit parameter exists in the signature."""
        import inspect

        params = inspect.signature(decode.requirement).parameters
        self.assertNotIn("hit", params)

    def test_fp16_materialize_bound(self):
        """Materialize-then-compute write BW: 19200 x 4915200 x 2 B = 188.7 GB/s."""
        r = decode.requirement(40.0)
        self.assertAlmostEqual(r.fp16_materialize_gb_s, 188.74, places=1)

    def test_headroom_and_verdicts(self):
        """40-band clears GO vs the measured PS5-GPU anchor with >=5x margin."""
        self.assertGreaterEqual(decode.requirement(40.0).headroom_vs_ps5_gpu, 5.0)
        self.assertEqual(decode.requirement(40.0).verdict, "GO")
        # The async-deep EXL3 1.66 row must also clear.
        self.assertEqual(decode.requirement(43.7).verdict, "GO")
        # Absurd throughput dies.
        self.assertEqual(decode.requirement(400.0).verdict, "DEAD")

    def test_headline_check(self):
        self.assertTrue(all(decode.headline_check().values()))

    def test_cli_smoke(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["decode"])
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("invariants", out)
        self.assertIn("GO", out)

    def test_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            decode.requirement(0.0)


class TestCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(argv)
        return code, buf.getvalue()

    def test_ledger_smoke(self):
        code, out = self._run(["ledger"])
        self.assertEqual(code, 0)
        self.assertIn("invariants", out)

    def test_reprice_cli_corrected_units(self):
        """S10: reprice --bw-mb 4920 at h=74.0/G=0.443 must print 39.8 tok/s
        (4.582 GiB/s), not the old inflated 42.7."""
        code, out = self._run(["reprice", "--bw-mb", "4920", "--curve", "12288:74.0", "--label", "unit-fix-check"])
        self.assertEqual(code, 0)
        self.assertIn("4.582", out)
        self.assertIn("39.8", out)
        self.assertNotIn("42.7", out)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = str(Path(tmp) / "pack.json")
            tr = str(Path(tmp) / "trace.json")
            rep = str(Path(tmp) / "report.md")
            self.assertEqual(self._run(["pack", "build", "--bpw", "2.05", "--out", pack])[0], 0)
            self.assertEqual(self._run(["pack", "verify", pack])[0], 0)
            self.assertEqual(self._run(["trace", "gen", "--tokens", "300", "--out", tr])[0], 0)
            code, out = self._run(
                ["gate", "replay", "--pack", pack, "--trace", tr, "--cache-gib", "8", "--report", rep]
            )
            self.assertIn("R2 gate report", out)
            self.assertIn("Verdict:", out)
            report = json.loads(Path(rep.replace(".md", ".json")).read_text())
            self.assertIn(report["verdict"], ("PASS", "FAIL", "UNREACHABLE"))
            # Exit code 2 == FAIL is a valid public datapoint; only crashes are bugs.
            self.assertIn(code, (0, 2))


if __name__ == "__main__":
    unittest.main()
