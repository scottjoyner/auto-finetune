"""Export orchestration tests, all artifacts are disposable fake data.

No model weights, external downloads, GPU execution, runtime deployment,
or irreversible conversion steps are permitted in this test module.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import model_export as exp


def test_gguf_missing_converter_fails_without_loading_any_model(monkeypatch,tmp_path):
    monkeypatch.setattr(exp.os.path, "exists", lambda _: False)
    monkeypatch.setattr(exp,"export_merged",
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError("merge forbidden")))
    result=exp.export_gguf("/absent/adapter",str(tmp_path/"output.gguf"))
    assert result.success is False
    assert "convert script not found" in result.error
    assert list(tmp_path.iterdir())==[]


def test_gguf_failed_merge_never_calls_conversion(monkeypatch,tmp_path):
    original_exists=exp.os.path.exists
    monkeypatch.setattr(exp.os.path,"exists",
                        lambda p: True if p=="/usr/local/bin/convert_hf_to_gguf.py" else original_exists(p))
    monkeypatch.setattr(exp,"export_merged",
                        lambda *a,**k: exp.ExportResult("merged",a[1],error="no verified base",success=False))
    monkeypatch.setattr(exp.subprocess,"run",
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError("must not convert")))
    result=exp.export_gguf("adapter",str(tmp_path/"output.gguf"))
    assert result.success is False
    assert result.error=="Merge failed: no verified base"


def test_gguf_success_uses_local_mock_only_and_cleans_temporary_merge(monkeypatch,tmp_path):
    original_exists=exp.os.path.exists
    converter="/usr/local/bin/convert_hf_to_gguf.py"
    monkeypatch.setattr(exp.os.path,"exists",
                        lambda p: True if p==converter else original_exists(p))
    out=tmp_path/"output.gguf"
    commands=[]
    def fake_merge(src,dst):
        Path(dst).mkdir()
        (Path(dst)/"fake-merged.txt").write_text("synthetic, not model weights")
        return exp.ExportResult("merged",dst,success=True)
    def fake_convert(command,**kwargs):
        commands.append((command,kwargs))
        out.write_bytes(b"FAKE-GGUF-TEST-DATA")
        return SimpleNamespace(returncode=0,stderr="")
    monkeypatch.setattr(exp,"export_merged",fake_merge)
    monkeypatch.setattr(exp.subprocess,"run",fake_convert)
    result=exp.export_gguf("/fake/adapter",str(out),"q8_0")
    assert result.success is True
    assert result.size_bytes==len(b"FAKE-GGUF-TEST-DATA")
    assert result.format=="gguf-q8_0"
    assert commands[0][0][1]==converter
    assert commands[0][1]["timeout"]==600
    assert not Path(str(out)+".merged").exists()


def test_gguf_converter_errors_are_reported(monkeypatch,tmp_path):
    original_exists=exp.os.path.exists
    converter="/usr/local/bin/convert_hf_to_gguf.py"
    monkeypatch.setattr(exp.os.path,"exists",
                        lambda p: True if p==converter else original_exists(p))
    monkeypatch.setattr(exp,"export_merged",
                        lambda src,dst: exp.ExportResult("merged",dst,success=True))
    monkeypatch.setattr(exp.subprocess,"run",
                        lambda *a,**k:SimpleNamespace(returncode=17,stderr="synthetic-conversion-failure"))
    result=exp.export_gguf("fake",str(tmp_path/"bad.gguf"))
    assert result.success is False
    assert result.error=="synthetic-conversion-failure"


def test_cli_dispatch_failure_and_success_are_explicit(monkeypatch,tmp_path,capsys):
    results=[]
    def fake_export(*a):
        results.append(a)
        return exp.ExportResult("merged",str(tmp_path/"fake"),success=True,size_bytes=50)
    monkeypatch.setattr(exp,"export_merged",fake_export)
    rc=exp.main(["--checkpoint","/fake","--output",str(tmp_path/"fake"),
                 "--format","merged","--base-model","toy"])
    assert rc==0
    assert results==[("/fake",str(tmp_path/"fake"),"toy")]
    assert "Success" in capsys.readouterr().out
    monkeypatch.setattr(exp,"export_onnx",
                        lambda *a: exp.ExportResult("onnx",str(tmp_path/"fake"),
                                                   success=False,error="no model"))
    rc=exp.main(["--checkpoint","fake","--output",str(tmp_path/"fake"),"--format","onnx"])
    assert rc==1
    assert "Failed: no model" in capsys.readouterr().out


def test_cli_gguf_dispatches_selected_quantization_without_real_export(monkeypatch,tmp_path,capsys):
    calls=[]
    def fake_export(source,dest,quantize):
        calls.append((source,dest,quantize))
        return exp.ExportResult("gguf-q4_0",dest,success=False,error="review gate closed")
    monkeypatch.setattr(exp,"export_gguf",fake_export)
    target=str(tmp_path/"sample.gguf")
    assert exp.main(["--checkpoint","/fake","--output",target,"--format","gguf","--quantize","q4_0"])==1
    assert calls==[("/fake",target,"q4_0")]
    assert "review gate closed" in capsys.readouterr().out
    assert not Path(target).exists()
