import json
from types import SimpleNamespace
import pytest
import benchmark_c1_all_models as benchmark


def config(tmp_path,key="candidate"):
    weights=tmp_path/'best.pt';weights.write_bytes(b'local-checkpoint')
    path=tmp_path/'models.json'
    path.write_text(json.dumps({'models':[{'key':key,'weights':str(weights)}]}))
    return path


def test_running_checkpoint_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark.torch,'load',lambda *a,**k:{'epoch':2,'optimizer':{},'model':SimpleNamespace(names={0:'enemy',1:'teammate'})})
    with pytest.raises(ValueError,match='not finalized'):
        benchmark.load_models(config(tmp_path))


def test_completed_checkpoint_requires_correct_identity_classes(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark.torch,'load',lambda *a,**k:{'epoch':-1,'optimizer':None,'model':SimpleNamespace(names={0:'person'})})
    with pytest.raises(ValueError,match='wrong classes'):
        benchmark.load_models(config(tmp_path))


def test_valid_checkpoint_has_resolved_path_and_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark.torch,'load',lambda *a,**k:{'epoch':-1,'optimizer':None,'model':SimpleNamespace(names={0:'enemy',1:'teammate'})})
    rows=benchmark.load_models(config(tmp_path))
    assert rows[0]['weights'].is_absolute()
    assert len(rows[0]['sha256'])==64
