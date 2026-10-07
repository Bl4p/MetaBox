"""The paper's experiment plan, run stage by stage. Finished runs are skipped, so it can be
stopped and restarted at any time.

    python -m l2t.experiments --list                  # show every stage and what is done
    python -m l2t.experiments --stage core            # one stage
    python -m l2t.experiments --stage all             # everything, in order

Agent labels: L2T-scratch = trained on the test set's own distribution (IV-B),
L2T = the BBOB_learn agent (IV-C), L2T-FT = BBOB_learn agent fine-tuned (IV-D).

Stages (paper section -> tables):
    core        IV-B/IV-E: DE agents per CEC17 set + BBOB9/10; vs STDE, fixed, random agents, DE peers
                -> Table S.V (G_roll), Table S.X (G_max), Fig. 3
    bbob_de     IV-C: DE agent on BBOB_learn (T = 5e6), tested on BBOB1-15 -> Table III (DE), S.VI (DE)
    bbob_ga     IV-C: GA agent on BBOB_learn (T = 5e6) vs GA peers -> Table III (GA), S.VI (GA)
    explicit    IV-C2: vs MTDE-EA and AT-MFEA on BBOB1-15 -> Table S.VIII
    finetune    IV-D: BBOB_learn DE agent fine-tuned on the 12 sets -> Table S.IX, Fig. 4
    ablation    IV-E: 8 retrained variants -> Table S.XI
    b2          IV-G: b2 in {0.1, 0.5, 1, 5, 10, 50, 100} on BBOB_learn -> Tables S.XII, S.XIII
    jade        IV-H3: MTJADE-L2T vs STJADE on the 12 sets -> Table S.XV
    rl_algos    IV-H2: SAC and TD3 agents on BBOB_learn vs the PPO agent -> Table S.XIV
Not covered: MFEA-RL (S.VII), HPO (S.XVI).
"""
import argparse
import os

from . import evaluate, train
from .problems import CEC_SETS

OUT = 'l2t_runs'
SCRATCH_SETS = list(CEC_SETS) + ['BBOB9', 'BBOB10']           # trained from scratch per set
BBOB_TEST = [f'BBOB{i}' for i in range(1, 16)]
ABLATION_SETS = list(CEC_SETS) + ['BBOB1', 'BBOB2', 'BBOB9', 'BBOB10']
FIXED = ['MTDE-f(.5,0,1)', 'MTDE-f(.5,1,0)', 'MTDE-f(.5,1,1)', 'MTDE-f(1,0,1)', 'MTDE-f(1,1,0)',
         'MTDE-f(1,1,1)', 'MTDE-r', 'STDE']
DE_PEERS = ['AEMTO', 'MFDE', 'MKTDE', 'MTDE-AD', 'MTDE-B']
GA_PEERS = ['GMFEA', 'MFEA', 'MFEA-AKT', 'MFEA2', 'MTEA-AD']


def run_dir(base, set_name, ablation=None, ft=False, b2=10.0):
    name = '_'.join(filter(None, [base, set_name, ablation, 'ft' if ft else None,
                                  None if b2 == 10 else f'b2_{b2:g}']))
    return os.path.join(OUT, name)


def ensure_trained(base, set_name, workers, ablation=None, init=None, b2=10.0, T=None):
    d = run_dir(base, set_name, ablation, init is not None, b2)
    if os.path.exists(os.path.join(d, 'done')):
        return d
    argv = ['--set', set_name, '--base', base, '--out', OUT, '--b2', str(b2)]
    argv += ['--workers', str(workers)] if workers else []
    argv += ['--ablation', ablation] if ablation else []
    argv += ['--init', init] if init else []
    argv += ['--T', str(T)] if T else []
    train.main(argv)
    open(os.path.join(d, 'done'), 'w').close()
    return d


def ensure_eval(sets, algos, base, workers):
    argv = ['--sets', *sets, '--algos', *algos, '--base', base, '--out', OUT]
    evaluate.main(argv + (['--workers', str(workers)] if workers else []))


def agent(label, d, gmax=False):
    """Agent spec for evaluate; gmax=True normalises time features by G_max (stored as <label>-gmaxnorm)."""
    path = os.path.join(d, "agent_best.pt")
    return f'{label}-gmaxnorm={path}:gmax' if gmax else f'{label}={path}'


def stage_core(w):
    for s in SCRATCH_SETS:
        d = ensure_trained('DE', s, w)
        ensure_eval([s], [agent('L2T-scratch', d), agent('L2T-scratch', d, gmax=True)] + FIXED + DE_PEERS, 'DE', w)


def stage_bbob_de(w):
    d = ensure_trained('DE', 'BBOB_learn', w, T=5e6)
    ensure_eval(BBOB_TEST, [agent('L2T', d), agent('L2T', d, gmax=True)] + DE_PEERS + ['STDE'], 'DE', w)
    ensure_eval(['BBOB1', 'BBOB2'], [agent('L2T', d)] + FIXED, 'DE', w)   # S.X rows BBOB1/2


def stage_bbob_ga(w):
    d = ensure_trained('GA', 'BBOB_learn', w, T=5e6)
    ensure_eval(BBOB_TEST, [agent('L2T', d), agent('L2T', d, gmax=True)] + GA_PEERS + ['STGA'], 'GA', w)


def stage_explicit(w):
    ensure_eval(BBOB_TEST, ['MTDE-EA'], 'DE', w)
    ensure_eval(BBOB_TEST, ['ATMFEA'], 'GA', w)


def stage_finetune(w):
    src = os.path.join(ensure_trained('DE', 'BBOB_learn', w, T=5e6), 'agent_best.pt')
    for s in SCRATCH_SETS:
        d = ensure_trained('DE', s, w, init=src)
        ensure_eval([s], [agent('L2T-FT', d), 'MTDE-EA'], 'DE', w)


def stage_ablation(w):
    from .env import L2TEnv
    for ab in L2TEnv.ABLATIONS:
        for s in SCRATCH_SETS:
            d = ensure_trained('DE', s, w, ablation=ab)
            ensure_eval([s], [agent(f'L2T-{ab}', d)], 'DE', w)
        d = ensure_trained('DE', 'BBOB_learn', w, ablation=ab, T=5e6)
        ensure_eval(['BBOB1', 'BBOB2'], [agent(f'L2T-{ab}', d)], 'DE', w)


def stage_b2(w):
    for b2 in (0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0):
        d = ensure_trained('DE', 'BBOB_learn', w, b2=b2, T=5e6)
        ensure_eval(BBOB_TEST, [agent(f'L2T-b2_{b2:g}', d)], 'DE', w)


def stage_jade(w):
    for s in SCRATCH_SETS:
        d = ensure_trained('JADE', s, w)
        ensure_eval([s], [agent('L2T', d), 'STJADE'], 'JADE', w)


def stage_rl_algos(w):
    from . import offpolicy
    for algo in ('SAC', 'TD3'):
        d = os.path.join(OUT, f'{algo}_BBOB_learn')
        if not os.path.exists(os.path.join(d, 'done')):
            offpolicy.main(['--algo', algo, '--set', 'BBOB_learn', '--out', OUT])
        ensure_eval(BBOB_TEST, [agent(algo, d)], 'DE', w)


STAGES = {   # order of --stage all: headline results first, the long ablation stage last
    'core': stage_core, 'bbob_de': stage_bbob_de, 'bbob_ga': stage_bbob_ga, 'explicit': stage_explicit,
    'finetune': stage_finetune, 'jade': stage_jade, 'rl_algos': stage_rl_algos, 'b2': stage_b2,
    'ablation': stage_ablation}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=list(STAGES) + ['all'])
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--list', action='store_true')
    args = p.parse_args(argv)
    if args.list or not args.stage:
        done = sorted(d for d in os.listdir(OUT) if os.path.exists(os.path.join(OUT, d, 'done'))) if os.path.isdir(OUT) else []
        print('stages:', ', '.join(STAGES))
        print('trained agents:', ', '.join(done) or 'none')
        return
    for name, fn in STAGES.items():
        if args.stage in (name, 'all'):
            print(f'==== stage {name}', flush=True)
            fn(args.workers)


if __name__ == '__main__':
    main()
