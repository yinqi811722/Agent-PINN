"""Run the supplied formal implementation from an explicit JSON configuration."""
from pathlib import Path
import argparse,json,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/formal_30seed.json')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    path=ROOT/('configs/smoke.json' if a.smoke else a.config)
    cfg=json.loads(path.read_text(encoding='utf8'))
    command=[sys.executable,str(ROOT/'src/run_direct_pinn_forecasting_30seeds_formal.py')]
    for key,value in cfg.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
        elif isinstance(value,list):command.extend([flag,*map(str,value)])
        else:command.extend([flag,str(value)])
    if a.cpu:command.append('--cpu')
    print(json.dumps(command,indent=2),flush=True)
    if not a.dry_run:
        env=os.environ.copy();env.setdefault('MPLBACKEND','Agg');env.setdefault('OMP_NUM_THREADS','1')
        env.setdefault('MKL_NUM_THREADS','1');env['PYTHONIOENCODING']='utf-8'
        raise SystemExit(subprocess.call(command,cwd=ROOT,env=env))
if __name__=='__main__':main()
