"""Supervise one local continuation and its fixed checkpoint evaluations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--outdir",type=Path,required=True)
    args=p.parse_args();source=args.source_run.resolve();out=args.outdir.resolve()
    if out.exists():raise FileExistsError(out)
    cfg=json.loads((source/"configuration.json").read_text())
    root=Path(__file__).resolve().parents[2]
    env={**os.environ,"VPML_JAX_BACKEND":"cpu","JAX_ENABLE_X64":"True","PYTHONUNBUFFERED":"1"}
    handles=[];workers=[]
    def launch(module,argv,log):
        handle=log.open("x");handles.append(handle)
        child=subprocess.Popen([sys.executable,"-u","-m",module,*map(str,argv)],cwd=root,env=env,
            stdout=handle,stderr=subprocess.STDOUT)
        workers.append(child);return child
    launch_log=out.with_name(out.name+".training.log")
    trainer=launch("model.diagnostics.hermite_linear_pilot",[
        "--outdir",out,"--reference-cache",cfg["reference_cache"],"--epochs",100,
        "--resume-from",source/"epoch005","--verify-resume-from",source/"epoch004"],launch_log)
    try:
        # Training writes this only after exact replay succeeds and E5 is saved.
        deadline=time.monotonic()+1800
        while not (out/"epoch005/rng_state.json").exists():
            if trainer.poll() is not None:raise RuntimeError(f"Training preflight exited {trainer.returncode}")
            if time.monotonic()>deadline:raise TimeoutError("Training preflight exceeded30 minutes")
            time.sleep(5)
        shutil.copy2(launch_log,out/"preflight.log")
        snapshots=out/"source";snapshots.mkdir()
        for f in root.glob("model/diagnostics/*hermite_linear*.py"):shutil.copy2(f,snapshots/f.name)
        evaluate=launch("model.diagnostics.evaluate_hermite_linear_pilot",[
            "--run",out,"--epochs","10,20,40,60,80,100","--wait-checkpoints","--eval-dt",.005,
            "--dealias","--skip-control","--name","evaluations_historical_settings"],out/"evaluation.log")
        validate=launch("model.diagnostics.validate_hermite_linear_pilot",[
            "--run",out,"--epochs","10,20,30,40,50,60,70,80,90,100"],out/"validation.log")
        (out/"supervisor.json").write_text(json.dumps(dict(supervisor_pid=os.getpid(),trainer_pid=trainer.pid,
            evaluator_pid=evaluate.pid,validator_pid=validate.pid,source_run=str(source),stop_epoch=100,
            training_log=str(launch_log)),indent=2))
        while any(child.poll() is None for child in workers):
            for child in workers:
                if child.poll() is not None and child.returncode!=0:
                    raise RuntimeError(f"Worker {child.pid} failed with exit {child.returncode}")
            time.sleep(10)
        reporter=launch("model.diagnostics.report_hermite_linear_continuation",["--run",out],out/"reporting.log")
        if reporter.wait()!=0:raise RuntimeError("Report generation failed")
        shutil.copy2(launch_log,out/"training.log")
        print(f"Complete: {out/'REPORT.md'}",flush=True)
    except BaseException as exc:
        for child in workers:
            if child.poll() is None:child.terminate()
        for child in workers:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill()
        if out.exists():(out/"FAILED.json").write_text(json.dumps(dict(error=repr(exc)),indent=2))
        raise
    finally:
        for h in handles:h.close()


if __name__=="__main__":main()
