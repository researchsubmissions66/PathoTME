"""Launch orchestration tests use fake Slurm responses, never real jobs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import launch_brca_vila as launcher


def test_completed_output_requires_matching_identity_and_artifacts(tmp_path):
    assert not launcher.complete(tmp_path,"id")
    (tmp_path/"metrics.json").write_text(json.dumps({"identity":"id","status":"completed","artifact_sha256":{}}))
    assert launcher.complete(tmp_path,"id")
    with pytest.raises(ValueError):launcher.complete(tmp_path,"different")


@pytest.mark.parametrize("submit",[False,True])
def test_panel_dependencies_and_dry_run(tmp_path,monkeypatch,submit):
    plans=[]
    for panel in ("m64","s62"):
        for kind in ("smoke","train"):
            plans.append({"panel":panel,"execution":kind,"job_name":panel+kind,
                          "dependency_panel":None if kind=="smoke" else panel,
                          "skip_completed":False,"components":[],"command_template":["sbatch","--parsable"]})
    monkeypatch.setattr(launcher,"ROOT",tmp_path)
    monkeypatch.setattr(launcher,"build_plan",lambda:plans)
    monkeypatch.setattr(launcher,"load_dotenv",lambda *a:None)
    monkeypatch.setattr(launcher.subprocess,"check_output",lambda *a,**k:"")
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout=str(100+len(calls)),stderr="")
    monkeypatch.setattr(launcher.subprocess,"run",fake)
    report=tmp_path/"report.json"
    monkeypatch.setattr(sys,"argv",["launcher","--report",str(report)]+(["--submit"] if submit else []))
    launcher.main()
    if submit:
        assert len(calls)==4
        assert "--dependency=afterok:101" in calls[1]
        assert "--dependency=afterok:103" in calls[3]
    else:assert not calls
    assert json.loads(report.read_text())["dry_run"] is not submit
