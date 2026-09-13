"""Run unchanged ViLa/MGPATH loops on new shot configs, with fixed prior repairs."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0]=['/path/to/PathoTME','/path/to/PGVL-Gym','/path/to/PathoTME/scripts']
from shot_contract import load_campaign,tag_for,validate_native,SHOTS
from pathotme.locked_tcga import load_launch,sha,verify_files


def selected(campaign,args):
    tag=tag_for(args.method,args.encoder)
    shots=SHOTS if args.smoke else [args.shots]
    result=[]
    for shot in shots:
        group=next(g for g in campaign['groups'] if (g['tag'],g['shots'])==(tag,shot))
        launch=load_launch(group['launch'])
        if launch['identity']!=group['identity']:raise ValueError('child launch identity changed')
        plan=next(p for p in launch['plans'] if (p['cohort'],p['method'],p['fold'])==(args.cohort,args.method,args.fold))
        if plan['encoder']!=args.encoder or plan['shots']!=shot:raise ValueError('pair/shot mismatch')
        result.append((launch,plan))
    return result


def execute(args,plans):
    import torch
    from common.configuration import load_dotenv
    import run_locked_tcga as native
    import run_cross_encoder_tcga as cross
    import pathotme.locked_models as factory
    from pathotme.brca_mgpath_index_repair import IndexedBreastGuidedMGPathMethod
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('allocated GPU job required')
    load_dotenv('/path/to/PGVL-Gym/.env')
    # Preserve the previously validated BRCA PLIP indexing repair in this
    # process only. Original factory and both training-loop files are unchanged.
    factory.BreastGuidedMGPathMethod=IndexedBreastGuidedMGPathMethod
    native.validate_native=validate_native
    cross.validate_native=validate_native
    # All eight completed 16-shot native fold-zero checkpoints are structural
    # fixtures only. The RN50 MGPATH port must not overwrite that checkpoint.
    def existing_fixture(plan,cfg,device):
        checkpoint=Path(plan['smoke_checkpoint_dir'])/f"fold{plan['fold']}_best.pt"
        verify_files({str(checkpoint):plan['smoke_checkpoint_sha256']})
        return {str(checkpoint):plan['smoke_checkpoint_sha256']}
    cross.smoke_fixture=existing_fixture
    runner=native if tag_for(args.method,args.encoder)=='native' else cross
    for launch,plan in plans:
        locks=Path(launch['output'])/'locks';locks.mkdir(exist_ok=True)
        with (locks/f"{args.cohort}_{args.method}_{'smoke' if args.smoke else args.fold}.lock").open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.smoke:
                from shot_smoke import run_smoke
                run_smoke(launch,plan,args.device)
            else:
                runner.execute(SimpleNamespace(smoke=False,device=args.device),launch,plan)
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--cohort',choices=['nsclc','brca'],required=True)
    parser.add_argument('--method',choices=['vila_mil','mgpath'],required=True)
    parser.add_argument('--encoder',choices=['plip','clip-rn50'],required=True)
    parser.add_argument('--shots',type=int,choices=SHOTS)
    parser.add_argument('--fold',type=int,choices=list(range(5)),required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.smoke and (args.fold!=0 or args.shots is not None):
        raise ValueError('one fold-zero smoke allocation covers both shot values')
    if not args.smoke and args.shots is None:raise ValueError('fold requires explicit shot count')
    campaign=load_campaign(args.campaign);plans=selected(campaign,args)
    if not args.execute:
        print(json.dumps({'campaign_identity':campaign['identity'],'smoke':args.smoke,
                          'plans':[p for l,p in plans]}));return
    execute(args,plans)


if __name__=='__main__':main()
