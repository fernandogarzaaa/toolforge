import json

import pytest

from toolforge.config import load_config
from toolforge.evaluate import bucket, load_eval_records, render_markdown, run_eval, score_records, slug, summarize

CALL = '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>'
TOOLS = [{"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}]


def rec(rid, category, source="bfcl-v4"):
    return {
        "id": rid,
        "source": source,
        "category": category,
        "tools": TOOLS,
        "messages": [{"role": "user", "content": f"q {rid}"}],
        "gold": [] if category == "irrelevance" else [{"f": {"a": [1]}}],
    }


def test_bucket_and_slug():
    assert bucket(rec("1", "simple")) == "bfcl/simple"
    assert bucket(rec("1", "irrelevance", source="glaive-function-calling-v2")) == "glaive/irrelevance"
    assert slug("sft+dpo") == "sft_dpo"
    assert slug("a/b c") == "a_b_c"


def test_load_eval_records_limits_per_bucket(tmp_path):
    f = tmp_path / "e.jsonl"
    rows = [rec(str(i), "simple") for i in range(5)] + [rec(f"g{i}", "simple", "glaive") for i in range(3)]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(load_eval_records([str(f)], None)) == 8
    limited = load_eval_records([str(f)], 2)
    assert [r["id"] for r in limited] == ["0", "1", "g0", "g1"]


def test_score_records_and_summary():
    recs = [rec("1", "simple"), rec("2", "simple"), rec("3", "irrelevance"), rec("4", "irrelevance")]
    outputs = {"1": CALL, "2": "no idea", "3": "Sorry, can't.", "4": CALL}
    rows = score_records(recs, outputs)
    assert [r["correct"] for r in rows] == [True, False, True, False]
    assert [r["reason"] for r in rows] == ["ok", "no_call", "ok", "spurious_call"]
    assert [r["did_call"] for r in rows] == [True, False, False, True]
    assert [r["should_call"] for r in rows] == [True, True, False, False]
    assert rows[0]["format"] == "tagged"
    s = summarize(rows, n_boot=100, seed=0)
    assert s["n"] == 4
    assert s["categories"]["bfcl/simple"]["mean"] == 0.5
    assert s["categories"]["bfcl/irrelevance"]["mean"] == 0.5
    assert s["macro_accuracy"] == 0.5
    assert s["micro_accuracy"]["mean"] == 0.5
    rel = s["relevance"]
    assert (rel["tp"], rel["fp"], rel["fn"], rel["tn"]) == (1, 1, 1, 1)
    assert rel["call_recall"] == 0.5
    assert rel["no_call_accuracy"] == 0.5
    assert s["error_reasons"] == {"bfcl/simple": {"no_call": 1}, "bfcl/irrelevance": {"spurious_call": 1}}


def test_macro_is_mean_of_categories_not_examples():
    recs = [rec("1", "simple"), rec("2", "simple"), rec("3", "simple"), rec("4", "irrelevance")]
    rows = score_records(recs, {"1": CALL, "2": CALL, "3": CALL, "4": CALL})
    s = summarize(rows, n_boot=50, seed=0)
    assert s["macro_accuracy"] == pytest.approx(0.5)
    assert s["micro_accuracy"]["mean"] == pytest.approx(0.75)


def test_summary_with_only_call_examples_has_nan_no_call_accuracy():
    rows = score_records([rec("1", "simple")], {"1": CALL})
    s = summarize(rows, n_boot=10, seed=0)
    assert s["relevance"]["call_recall"] == 1.0
    assert s["relevance"]["no_call_accuracy"] != s["relevance"]["no_call_accuracy"]  # NaN


def test_render_markdown_empty():
    assert render_markdown({"models": {}, "n_examples": 0}) == "No completed model evaluations.\n"


def _write_preds(out_dir, name, outputs):
    p = out_dir / "predictions" / f"{slug(name)}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps({"id": k, "output": v}) for k, v in outputs.items()) + "\n")


def test_run_eval_from_saved_predictions(tmp_path):
    recs = [rec("1", "simple"), rec("2", "irrelevance")]
    f = tmp_path / "e.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs))
    out = tmp_path / "res"
    _write_preds(out, "base", {"1": "nothing", "2": CALL})
    _write_preds(out, "sft", {"1": CALL, "2": "Sorry"})
    _write_preds(out, "partial", {"1": CALL})
    cfg = load_config(
        None,
        [
            f"eval.files=[{f}]",
            f"eval.out_dir={out}",
            "eval.n_bootstrap=50",
            "eval.models=[{name: base}, {name: sft}, {name: partial}, {name: missing}]",
        ],
    )
    report = run_eval(cfg, only=["__none__"])
    assert list(report["models"]) == ["base", "sft"]
    assert report["models"]["base"]["macro_accuracy"] == 0.0
    assert report["models"]["sft"]["macro_accuracy"] == 1.0
    diff = report["paired_vs_first"]["sft - base"]
    assert diff["mean"] == 1.0
    md = (out / "RESULTS.md").read_text()
    assert "| bfcl/simple | 1 | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |" in md
    assert "sft - base: +100.0" in md
    assert json.loads((out / "summary.json").read_text())["n_examples"] == 2
    scored = (out / "scored" / "sft.jsonl").read_text().splitlines()
    assert len(scored) == 2
