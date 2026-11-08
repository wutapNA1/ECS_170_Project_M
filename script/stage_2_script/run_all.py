'''
Stage 2 - run the whole pipeline with one click.

This script calls the `main()` of the four Stage 2 scripts one after another, so the
complete "train -> test -> draw the learning curves -> summarise" workflow of Stage 2
can be started from a single PyCharm run configuration:

    python script/stage_2_script/run_all.py             # run all four steps
    python script/stage_2_script/run_all.py mlp dt      # run only the given steps

The names of the steps are mlp, dt, svm and load. Every step prints its own output and
the total training time of each step is reported at the end.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import importlib
import os
import sys
import time
import traceback

# make the `local_code` package importable no matter which folder this script is started from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# the four scripts live next to this file, so they can be imported by their module name
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch

# ---- the steps of the pipeline, in the order they have to run -------------------------
# (short name used on the command line, module name, description)
STEPS = [
    ('mlp', 'script_mlp', 'MLP experiments of task 2-3 and 2-4'),
    ('dt', 'script_decision_tree', 'decision tree on the same data set'),
    ('svm', 'script_svm', 'support vector machine on the same data set'),
    ('load', 'script_load_result', 'reload the saved results and summarise them'),
]


def main():
    print('=' * 88)
    print('Stage 2 - the complete pipeline')
    print('python interpreter: ' + sys.executable)
    print('torch: ' + torch.__version__ + ', GPU available: ' + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('GPU: ' + torch.cuda.get_device_name(0))
    print('project root: ' + PROJECT_ROOT)
    print('=' * 88)

    # `python run_all.py mlp dt` runs only the selected steps, no argument runs every step
    selected = [name.lower() for name in sys.argv[1:]]
    known = [step[0] for step in STEPS]
    unknown = [name for name in selected if name not in known]
    if unknown:
        print('unknown step(s): ' + ', '.join(unknown))
        print('available steps: ' + ', '.join(known))
        return 1
    steps = [step for step in STEPS if not selected or step[0] in selected]

    results = []
    failed = None
    started_all = time.time()
    for name, module_name, description in steps:
        print('')
        print('#' * 88)
        print('# step ' + name + ': ' + description)
        print('#' * 88)
        started = time.time()
        try:
            module = importlib.import_module(module_name)
            module.main()
            results.append((name, description, time.time() - started, 'ok'))
        except Exception:
            elapsed = time.time() - started
            results.append((name, description, elapsed, 'FAILED'))
            print('')
            print('step ' + name + ' failed after ' + str(round(elapsed, 1)) + 's:')
            traceback.print_exc()
            failed = name
            break

    # ---- summary of the run ----------------------------------------------------------
    print('')
    print('=' * 88)
    print('%-6s %-52s %10s  %s' % ('step', 'description', 'time', 'status'))
    for name, description, elapsed, status in results:
        print('%-6s %-52s %9.1fs  %s' % (name, description, elapsed, status))
    total = time.time() - started_all
    print('total: ' + str(round(total, 1)) + 's')
    if failed:
        print('the pipeline stopped at step ' + failed + ', see the traceback above')
    else:
        print('everything finished, the results are in result/stage_2_result')
    print('=' * 88)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
