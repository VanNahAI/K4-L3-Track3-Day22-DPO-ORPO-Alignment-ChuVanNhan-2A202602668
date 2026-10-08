"""CPU-only structural checks: sources parse, trainer APIs match TRL 1.13 /
transformers 5, and the Colab bundles are in sync with the sources.

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "00_dpo_loss_from_scratch", "01_sft_mini", "02_preference_data", "03_dpo_train",
    "03b_dpo_variants", "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
    "07_grpo_bonus",
]
SOURCES = [REPO / "notebooks" / f"{nb}.py" for nb in NOTEBOOKS] + sorted((REPO / "scripts").glob("*.py")) + sorted(
    (REPO / "lab22").glob("*.py")
)


def test_sources_exist_and_parse():
    for p in SOURCES:
        assert p.exists(), f"missing {p}"
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


def test_no_removed_trainer_arguments():
    # tokenizer= (TRL >= 0.13), warmup_ratio (transformers 5), max_prompt_length (TRL 1.x DPOConfig).
    banned = re.compile(r"\btokenizer\s*=\s*tokenizer\b|\bwarmup_ratio\s*=|\bmax_prompt_length\s*=")
    offenders = [str(p.relative_to(REPO)) for p in SOURCES if banned.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"removed trainer arguments in {offenders}"


def test_no_hardcoded_judge_model():
    for p in SOURCES:
        if p.name == "test_smoke.py":
            continue
        text = p.read_text(encoding="utf-8")
        assert "gpt-4o-mini" not in text and "claude-haiku" not in text, f"hard-coded judge id in {p}"


def test_kaggle_bundle_preserves_student_code():
    from build_kaggle import SOURCE, TARGET, render, validate

    nb = render()
    validate(nb)
    assert json.loads(TARGET.read_text(encoding="utf-8")) == nb
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    original_code = ["".join(c["source"]) for c in original["cells"][5:] if c["cell_type"] == "code"]
    kaggle_code = ["".join(c["source"]) for c in nb["cells"][5:]
                   if c["cell_type"] == "code" and not "".join(c["source"]).startswith("%pip")]
    normalized = [s.replace('        device_map={"": 0},\n', '')
                  if s.startswith("%%writefile /kaggle/working/lab22/lab22/modeling.py") else s
                  for s in kaggle_code]
    assert normalized == [s.replace("/content/lab22", "/kaggle/working/lab22") for s in original_code]
    assert any('        device_map={"": 0},\n' in s for s in kaggle_code)
    setup = "".join(nb["cells"][2]["source"])
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in setup
    assert 'os.environ["COMPUTE_TIER"] = "T4"' in setup
    core_install = "".join(nb["cells"][3]["source"])
    assert "unsloth" in core_install
    assert all(dep not in core_install for dep in ("llama-cpp-python", "lm-eval", "openai", "anthropic"))
    work_setup = "".join(nb["cells"][4]["source"])
    assert work_setup.index("import unsloth") < work_setup.index("import torch")
    assert "raise FileExistsError" in work_setup
    assert "source.is_relative_to" in work_setup


def test_colab_bundles_are_valid_and_current():
    from build_colab import render

    for tier, path in (("T4", "Lab22_DPO_T4.ipynb"), ("BIGGPU", "Lab22_DPO_BigGPU.ipynb")):
        on_disk = json.loads((REPO / "colab" / path).read_text(encoding="utf-8"))
        assert on_disk == render(tier), f"colab/{path} is stale: run `make colab`"
