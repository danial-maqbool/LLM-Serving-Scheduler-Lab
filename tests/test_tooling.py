from pathlib import Path
import json
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_design_generator_matches_committed_config():
    build=runpy.run_path(str(ROOT/'scripts/make_suite.py'))['build_suite']
    assert build()==json.loads((ROOT/'configs/study.json').read_text())


def test_markdown_local_links_and_anchors(tmp_path):
    check=runpy.run_path(str(ROOT/'scripts/check_links.py'))['check']
    (tmp_path/'README.md').write_text('[Method](method.md#explicit-assumptions)\n')
    (tmp_path/'method.md').write_text('# Explicit assumptions\n')
    assert check(tmp_path)==1
    (tmp_path/'README.md').write_text('[Broken](missing.md)')
    with pytest.raises(ValueError,match='missing'):check(tmp_path)


def test_report_refuses_partial_study(tmp_path):
    from llm_scheduler_lab.experiments import run_suite
    build=runpy.run_path(str(ROOT/'scripts/build_report.py'))['build']
    out=tmp_path/'smoke'
    run_suite(ROOT/'configs/smoke.json',out,allow_dirty=True,quiet=True)
    with pytest.raises(ValueError,match='955'):build(out,tmp_path/'report')


def test_release_checker_rejects_dirty_results(tmp_path):
    import llm_scheduler_lab.experiments as e
    check=runpy.run_path(str(ROOT/'scripts/check_artifacts.py'))['check']
    out=tmp_path/'out'
    # Inject declared provenance, not a dependency on the test runner's worktree state.
    from unittest.mock import patch
    with patch.object(e,'git_state',return_value={'git_commit':'test','git_dirty':True}):
        e.run_suite(ROOT/'configs/smoke.json',out,allow_dirty=True,quiet=True)
    with pytest.raises(ValueError,match='clean source'):check(out)
