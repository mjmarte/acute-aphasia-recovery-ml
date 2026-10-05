#!/usr/bin/env python3
"""Runs a list of analyses in sequence."""
import sys, os, subprocess, time, json
from pathlib import Path
import sys as _s; PY = _s.executable
HERE = Path(__file__).parent; OUT = Path(os.environ.get('REV_ROOT', os.environ.get('REV_ROOT', '.'))) / 'experiments' / 'models'
FS = ['FS1', 'FS2', 'FS3', 'FS4']; MODELS = ['ridge', 'svr', 'rf', 'xgb']
def cells(variant, fss=FS, models=MODELS): return [(variant, f, m) for f in fss for m in models]
QUEUES = {
 'A': cells('wab_base', FS, ['ridge', 'svr']) + cells('wab_base', ['FS4', 'FS1'], ['rf']) + cells('wab_base', ['FS4', 'FS1'], ['xgb'])
      + cells('c_wab_own', ['FS1', 'FS4'], ['ridge', 'svr']) + cells('c_wab_cu', ['FS1', 'FS4'], ['ridge', 'svr']) + cells('c_wab_both', ['FS1', 'FS4'], ['ridge', 'svr'])
      + cells('sub_wab', ['FS1', 'FS4'], ['ridge', 'svr']) + cells('res_wab', ['FS1', 'FS4'], ['ridge', 'svr'])
      + cells('wab_base', ['FS2', 'FS3'], ['rf', 'xgb'])
      + cells('c_wab_own', ['FS4', 'FS1'], ['rf']) + cells('c_wab_cu', ['FS4', 'FS1'], ['rf']) + cells('c_wab_both', ['FS4', 'FS1'], ['rf'])
      + cells('sub_wab', ['FS4', 'FS1'], ['rf']) + cells('res_wab', ['FS4', 'FS1'], ['rf'])
      + cells('c_wab_own', ['FS1', 'FS4'], ['xgb']) + cells('c_wab_cu', ['FS1', 'FS4'], ['xgb']) + cells('c_wab_both', ['FS1', 'FS4'], ['xgb']),
 'B': cells('nct_base', FS, ['ridge', 'svr']) + cells('nct_base', ['FS4', 'FS1'], ['rf']) + cells('nct_base', ['FS4', 'FS1'], ['xgb'])
      + cells('c_nct_own', ['FS1', 'FS4'], ['ridge', 'svr']) + cells('c_nct_aq', ['FS1', 'FS4'], ['ridge', 'svr']) + cells('c_nct_both', ['FS1', 'FS4'], ['ridge', 'svr'])
      + cells('red_aq', FS, ['ridge', 'svr']) + cells('red_own', ['FS1', 'FS4'], ['ridge', 'svr']) + cells('res_nct', ['FS1', 'FS4'], ['ridge', 'svr'])
      + cells('nct_base', ['FS2', 'FS3'], ['rf', 'xgb'])
      + cells('c_nct_own', ['FS4', 'FS1'], ['rf']) + cells('c_nct_aq', ['FS4', 'FS1'], ['rf']) + cells('c_nct_both', ['FS4', 'FS1'], ['rf'])
      + cells('res_nct', ['FS4', 'FS1'], ['rf']) + cells('red_aq', ['FS4', 'FS1'], ['rf']) + cells('red_own', ['FS4', 'FS1'], ['rf'])
      + cells('c_nct_own', ['FS1', 'FS4'], ['xgb']) + cells('c_nct_aq', ['FS1', 'FS4'], ['xgb']) + cells('c_nct_both', ['FS1', 'FS4'], ['xgb']),
}
if __name__ == '__main__':
    q = sys.argv[1]; log = open(OUT / f'queue_{q}.log', 'a')
    for variant, fs, model in QUEUES[q]:
        if (OUT / variant / f'task_{fs}_{model}.json').exists():
            continue
        t0 = time.time(); log.write(f"{time.strftime('%H:%M:%S')} START {variant} {fs} {model}\n"); log.flush()
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', LOKY_MAX_CPU_COUNT='6')
        r = subprocess.run([PY, str(HERE / 'run_model.py'), '--variant', variant, '--fs', fs, '--model', model], capture_output=True, text=True, env=env)
        tail = (r.stdout.strip().splitlines() or [''])[-1]
        log.write(f"{time.strftime('%H:%M:%S')} {'OK' if r.returncode == 0 else 'FAIL'} {variant} {fs} {model} {(time.time()-t0)/60:.1f} min :: {tail}\n")
        if r.returncode != 0: log.write(r.stderr[-2000:] + '\n')
        log.flush()
    log.write(f"{time.strftime('%H:%M:%S')} QUEUE {q} DONE\n"); log.close()
