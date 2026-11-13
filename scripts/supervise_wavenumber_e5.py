"""Observe the active reference controller and recover this one approved E5 lineage."""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/Users/armin/Documents/NYU/vpml-generalization')
OUT=Path('/Users/armin/Documents/NYU/vpml/out_bench/wavenumber_generalization_v2_e5_20260925')
SOURCE='ad8cabb58635341993b24385b550f950f6d7398c69c7193b50ea820c4b19e88e'
MODAL_SCRIPT='model/train/modal_wavenumber_e5.py'
CAP_DOLLARS=350.0
STOP_DOLLARS=330.0


def record(name,payload):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),**payload),indent=2)+'\n')
    os.replace(tmp,path)


def cost():
    result=subprocess.run(['modal','billing','report','--for','today','--json'],
                          cwd=ROOT,capture_output=True,text=True,check=True)
    entries=json.loads(result.stdout)
    return sum(float(x['Cost']) for x in entries
               if x.get('Description')=='vpml-wavenumber-generalization-v2-e5')


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def guard_budget():
    amount=cost()
    record('budget_status.json',dict(cost_reported_dollars=amount,
                                     stop_threshold_dollars=STOP_DOLLARS,
                                     authorized_cap_dollars=CAP_DOLLARS))
    return amount<STOP_DOLLARS


def await_process(pid):
    while alive(pid):
        if not guard_budget():
            os.kill(pid,signal.SIGTERM)
            raise RuntimeError('Budget stop while reference controller was running')
        time.sleep(300)


def invoke(label,args):
    logfile=OUT/f'{label}.log'
    with logfile.open('a') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True)
        record('supervisor_status.json',dict(stage=label,pid=process.pid,log=str(logfile)))
        while process.poll() is None:
            if not guard_budget():
                process.terminate()
                raise RuntimeError(f'Budget stop during {label}')
            time.sleep(300)
        if process.returncode:
            raise RuntimeError(f'{label} exited {process.returncode}; inspect {logfile}')


def completed_references():
    manifest=json.loads((OUT/'manifest.json').read_text())
    return sum((OUT/'reference'/c['case_id']/'states.npy').exists() for c in manifest['cases'])


def main(initial_pid):
    record('supervisor_status.json',dict(stage='watch_existing_reference_controller',pid=initial_pid))
    await_process(initial_pid)
    base=['modal','run',f'{MODAL_SCRIPT}::run_references','--source-sha',SOURCE,
          '--local-root',str(OUT)]
    attempts=0
    while completed_references()<75:
        attempts+=1
        if attempts>3:raise RuntimeError('Reference controller failed three recovery attempts')
        if not guard_budget():raise RuntimeError('Budget stop before reference resume')
        invoke('references_resume',base)
    if not guard_budget():raise RuntimeError('Budget stop before E5 training')
    train=['modal','run',f'{MODAL_SCRIPT}::run_training','--source-sha',SOURCE,
           '--local-root',str(OUT),'--stop-epoch','5']
    attempts=0
    while not (OUT/'evaluation/epoch_0005/report.json').exists():
        attempts+=1
        if attempts>3:raise RuntimeError('E5 trainer failed three recovery attempts')
        invoke('training_E5',train)
    plot=[str(Path('/Users/armin/Documents/NYU/vpml/.venv/bin/python')),
          'scripts/plot_wavenumber_e5.py','--run',str(OUT)]
    subprocess.run(plot,cwd=ROOT,check=True)
    record('COMPLETE.json',dict(stage='complete',references=75,
                                E5_report=str(OUT/'evaluation/epoch_0005/report.json')))


if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception as exc:
        record('FAILED.json',dict(error=repr(exc)))
        raise
