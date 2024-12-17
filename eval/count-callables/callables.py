#!/usr/bin/env python3

import os
import sys
import time
import queue
import argparse
import importlib
import contextlib
import multiprocessing
import multiprocessing.connection
import subprocess
import glob
import shutil
from dataclasses import dataclass
from collections import OrderedDict

import qwerty

benchmarks_dir = 'benchmarks/'
benchmarks_src = os.path.join(os.path.dirname(__file__), os.path.dirname(benchmarks_dir))

# ------------------------------------Running Algorithms----------------------------------------
def dj_balanced_circs():
    return [
        ('dj_balanced', 'Qwerty-Opt', 'Deutsch-Jozsa', 'Balanced'),
        ('dj_balanced', 'Qwerty-No-Opt', 'Deutsch-Jozsa', 'Balanced'),
        ('dj_balanced', 'Qsharp', 'Deutsch-Jozsa', 'Balanced'),
    ]
def dj_constant_circs():
    return [
        ('dj_constant', 'Qwerty-Opt', 'Deutsch-Jozsa', 'Constant'),
        ('dj_constant', 'Qwerty-No-Opt', 'Deutsch-Jozsa', 'Constant'),
        ('dj_constant', 'Qsharp', 'Deutsch-Jozsa', 'Constant'),
    ]
def bv_circs():
    return [
        ('bv', 'Qwerty-Opt', 'Bernstein-Vazirani', 'Alternating'),
        ('bv', 'Qwerty-No-Opt', 'Bernstein-Vazirani', 'Alternating'),
        ('bv', 'Qsharp', 'Bernstein-Vazirani', 'Alternating'),
    ]
def grover_circs():
    return [
        ('grover', 'Qwerty-Opt', 'Grover', 'All Ones'),
        ('grover', 'Qwerty-No-Opt', 'Grover', 'All Ones'),
        ('grover', 'Qsharp', 'Grover', 'All Ones'),
    ]
def simon_circs():
    return [
        ('simon', 'Qwerty-Opt', 'Simon', 'Secret Sauce'),
        ('simon', 'Qwerty-No-Opt', 'Simon', 'Secret Sauce'),
        ('simon', 'Qsharp', 'Simon', 'Secret Sauce'),
    ]
def period_circs():
    return [
        ('period', 'Qwerty-Opt', 'Period', 'Masking'),
        ('period', 'Qwerty-No-Opt', 'Period', 'Masking'),
        ('period', 'Qsharp', 'Period', 'Masking'),
    ]

def get_all_circs(arg_algo):
    algo_circ_funcs = OrderedDict([
        ('dj_constant', dj_constant_circs),
        ('dj_balanced', dj_balanced_circs),
        ('bv', bv_circs),
        ('grover', grover_circs),
        ('simon', simon_circs),
        ('period', period_circs),
    ])
    circs = []

    if arg_algo in algo_circ_funcs:
        circs.extend(algo_circ_funcs[arg_algo]())
    elif arg_algo == 'all':
        for circ_func in algo_circ_funcs.values():
            circs.extend(circ_func())
    else:
        raise ValueError('Invalid algorithm name')

    return circs

# --------------------------------------------------------------------------------------------------
def count_callables(qir):
    counts = [0]*9
    with open(qir, "r") as f:
        for line in f:
            linelst = line.split()
            func = None
            if(len(linelst) > 0 and linelst[0] == "call" and "(" in line):
                func = linelst[2]
            elif(len(linelst) > 2 and linelst[2] == "call" and "(" in line):
                func = linelst[4]
            
            if(func != None):
                if("__quantum__rt__callable_create" in func):
                    counts[0] += 1
                elif("__quantum__rt__callable_copy" in func):
                    counts[1] += 1
                elif("__quantum__rt__callable_invoke" in func):
                    counts[2] += 1
                elif("__quantum__rt__callable_make_adjoint" in func):
                    counts[3] += 1
                elif("__quantum__rt__callable_make_controlled" in func):
                    counts[4] += 1
                elif("__quantum__rt__callable_update_reference_count" in func):
                    counts[5] += 1
                elif("__quantum__rt__callable_update_alias_count" in func):
                    counts[6] += 1
                elif("__quantum__rt__capture_update_reference_count" in func):
                    counts[7] += 1
                elif("__quantum__rt__capture_update_alias_count" in func):
                    counts[8] += 1
    
    return counts

def evaluate_qwerty_no_opt(circ, algo_dir):
    algo_ugly, lang_pretty, algo_pretty, params = circ
    mycwd = os.getcwd()
    qir = algo_ugly + "_qwerty_no_opt.ll"
    re_res = []

    # Compile Qwerty's version of Algorithm then translate it to QIR
    try:
        os.chdir(algo_dir)
        sys.path.append('.')
        importlib.import_module(algo_ugly+"_no_opt")
        with open(qir, 'w') as fp:
            fp.write(qwerty.get_qir())

        result = count_callables(qir)
        create = result[0]
        copy = result[1]
        invoke = result[2]
        adjoint = result[3]
        controlled = result[4]
        callableRef = result[5]
        callableAlias = result[6]
        captureRef = result[7]
        captureAlias = result[8]
        re_res.append(f'{lang_pretty}, {algo_pretty}, {params}, {create}, {copy}, {invoke}, {adjoint}, {controlled}, {callableRef}, {callableAlias}, {captureRef}, {captureAlias}\n')     
    
        for p in glob.glob("*.dot"):
            os.remove(p)
        
        for p in glob.glob("*.txt"):
            os.remove(p)

        for p in glob.glob("*.mlir"):
            os.remove(p)
        
        for p in glob.glob("*.pyast"):
            os.remove(p)
    except subprocess.CalledProcessError as e:
        re_res.append(f'{lang_pretty}, {algo_pretty}, {params}, Failed\n')

    os.chdir(mycwd)
    return re_res


def evaluate_qwerty(circ, algo_dir):
    algo_ugly, lang_pretty, algo_pretty, params = circ
    mycwd = os.getcwd()
    qir = algo_ugly + "_qwerty.ll"
    re_res = []

    # Compile Qwerty's version of Algorithm then translate it to QIR
    try:
        os.chdir(algo_dir)
        sys.path.append('.')
        importlib.import_module(algo_ugly)
        with open(qir, 'w') as fp:
            fp.write(qwerty.get_qir())

        result = count_callables(qir)
        create = result[0]
        copy = result[1]
        invoke = result[2]
        adjoint = result[3]
        controlled = result[4]
        callableRef = result[5]
        callableAlias = result[6]
        captureRef = result[7]
        captureAlias = result[8]
        re_res.append(f'{lang_pretty}, {algo_pretty}, {params}, {create}, {copy}, {invoke}, {adjoint}, {controlled}, {callableRef}, {callableAlias}, {captureRef}, {captureAlias}\n')     
    
        for p in glob.glob("*.dot"):
            os.remove(p)
        
        for p in glob.glob("*.txt"):
            os.remove(p)

        for p in glob.glob("*.mlir"):
            os.remove(p)
        
        for p in glob.glob("*.pyast"):
            os.remove(p)
    except subprocess.CalledProcessError as e:
        re_res.append(f'{lang_pretty}, {algo_pretty}, {params}, Failed\n')

    os.chdir(mycwd)
    return re_res

def evaluate_qsharp(circ, algo_dir):
    algo_ugly, lang_pretty, algo_pretty, params = circ
    mycwd = os.getcwd()
    qir = "qir/" + algo_ugly + ".ll"
    qir_new = algo_ugly + "_qsharp.ll"
    re_res = []

    # Compile Qwerty's version of Algorithm then translate it to QIR
    try:
        os.chdir(algo_dir)
        subprocess.check_output(["dotnet build"], shell=True)
        shutil.copyfile(qir, qir_new)

        result = count_callables(qir)
        create = result[0]
        copy = result[1]
        invoke = result[2]
        adjoint = result[3]
        controlled = result[4]
        callableRef = result[5]
        callableAlias = result[6]
        captureRef = result[7]
        captureAlias = result[8]
        re_res.append(f'{lang_pretty}, {algo_pretty}, {params}, {create}, {copy}, {invoke}, {adjoint}, {controlled}, {callableRef}, {callableAlias}, {captureRef}, {captureAlias}\n')
    
        shutil.rmtree("bin")
        shutil.rmtree("obj")
        shutil.rmtree("qir")
    except subprocess.CalledProcessError as e:
        re_res.append(f'{lang_pretty}, {algo_pretty}, {params}, Failed\n')

    os.chdir(mycwd)
    return re_res

@dataclass
class AlgoDirs:
    algo_dir: str

def make_algo_dir(algo_ugly):
    # Create Algo Folder in results directory
    algo_folder = os.path.join(benchmarks_src, algo_ugly)
    if not os.path.exists(algo_folder):
        os.makedirs(algo_folder)

    return AlgoDirs(algo_folder)

@dataclass
class Config:
    all_algo_dirs: dict[str, AlgoDirs]

def get_all_tasks(*args, **kwargs):
    return get_all_circs(*args, **kwargs)

def run_task(circ, cfg):
    ugly_name = circ[0]
    lang = circ[1]
    algo_dirs = cfg.all_algo_dirs[ugly_name]

    if(lang == "Qsharp"):
        return evaluate_qsharp(circ, algo_dirs.algo_dir)

    if(lang == "Qwerty-No-Opt"):
        return evaluate_qwerty_no_opt(circ, algo_dirs.algo_dir)

    return evaluate_qwerty(circ, algo_dirs.algo_dir)

def run_process(out_queue, task_idx, all_tasks_args, cfg):
    tasks = get_all_tasks(*all_tasks_args)
    task = tasks[task_idx]
    ret = run_task(task, cfg)
    out_queue.put_nowait(ret)

# Took this multiprocessing code from qre.py
@dataclass
class RunningProcess:
    proc: multiprocessing.Process
    queue: multiprocessing.Queue
    circ_idx: int
    start_sec: float

def run_parallel_tasks(timeout_sec, n_tasks, all_tasks_args, cfg):
    ctx = multiprocessing.get_context()
    nproc = multiprocessing.cpu_count()
    procs = [None]*nproc
    in_queue = list(reversed(range(n_tasks)))

    while True:
        done = True
        earliest_start_sec = None

        # First step: kickoff new processes in empty slots
        for proc_idx, running_proc in enumerate(procs):
            if running_proc is None:
                if in_queue:
                    done = False
                    next_task_idx = in_queue.pop()
                    out_queue = ctx.Queue()
                    proc_args = (out_queue, next_task_idx, all_tasks_args, cfg)
                    new_proc = ctx.Process(target=run_process,
                                           args=proc_args,
                                           daemon=True)
                    new_proc.start()
                    procs[proc_idx] = RunningProcess(
                        new_proc, out_queue, next_task_idx,
                        time.monotonic())
            else:
                done = False
                if earliest_start_sec is None \
                        or running_proc.start_sec < earliest_start_sec:
                    earliest_start_sec = running_proc.start_sec

        if done:
            break

        now = time.monotonic()
        if earliest_start_sec is None:
            wait_for = timeout_sec
        elif (biggest_delta := now - earliest_start_sec) < timeout_sec:
            # The check above is to ensure that the calculation below does
            # not result in a negative wait time
            wait_for = timeout_sec - biggest_delta
        else:
            # ...if it did, then wait for a couple of seconds instead of
            # waiting for 0 seconds, which would amount to burning some
            # coal in Macon to fire off a bunch of unnecessary syscalls
            wait_for = 2.0
        ready = set(multiprocessing.connection.wait(
            [proc.proc.sentinel for proc in procs if proc is not None],
            timeout=wait_for))
        now = time.monotonic()

        for proc_idx, proc in enumerate(procs):
            if proc is None:
                continue

            if proc.proc.sentinel in ready:
                try:
                    # Wait a couple of seconds just in case of weird
                    # coherence trolling or something
                    result = proc.queue.get(block=True, timeout=2.0)
                except queue.Empty:
                    raise ValueError('process finished without a result. '
                                     'data is missing')

                assert proc.queue.empty()
                proc.queue.close()
                proc.proc.join(timeout=2.0)
                if proc.proc.exitcode is None:
                    raise ValueError('Join is blocking after sentinel was '
                                     'ready, how?')
                proc.proc.close()
                procs[proc_idx] = None

                yield (proc.circ_idx, result)
            elif now - proc.start_sec > timeout_sec \
                    and proc.proc.is_alive():
                proc.queue.close()
                proc.proc.terminate()
                proc.proc.join(timeout=2.0)
                if proc.proc.exitcode is None:
                    raise ValueError('Join is blocking after we sent '
                                     'SIGTERM, how?')
                proc.proc.close()
                procs[proc_idx] = None

                yield (proc.circ_idx, None)

def main():
    parser = argparse.ArgumentParser(description='Resource Estimator')

    parser.add_argument(
        '-a',
        '--algorithm',
        type = str,
        default = "all",
        help='type the name of the algorithm you want to test over.'
    )
    parser.add_argument(
        '-t',
        '--timeout',
        type=float,
        default=2*60*60,
        help='Per-job timeout in seconds (default: 2 hours)'
    )
    args = parser.parse_args()

    mycwd = os.path.dirname(__file__)
    os.chdir(mycwd)

    get_all_circs_args = (args.algorithm.lower(),)
    circs = get_all_circs(*get_all_circs_args)

    ugly_names = {c[0] for c in circs}
    algo_dirs = {ugly_name: make_algo_dir(ugly_name) for ugly_name in ugly_names}
    
    with contextlib.ExitStack() as exit_stack:
        algo_csvs = {ugly_name: (exit_stack.enter_context(
                                     open(os.path.join(algo_dirs[ugly_name].algo_dir,
                                                       'callables_count.csv'), 'w')))
                     for ugly_name in ugly_names}

        for csvs in algo_csvs.values():
            re_csv = csvs
            re_csv.write(f'Language, Algorithm, Type, __quantum__rt__callable_create, __quantum__rt__callable_copy, __quantum__rt__callable_invoke, __quantum__rt__callable_make_adjoint, __quantum__rt__callable_make_controlled, __quantum__rt__callable_update_reference_count, __quantum__rt__callable_update_alias_count, __quantum__rt__capture_update_reference_count, __quantum__rt__capture_update_alias_count\n')

        cfg = Config(algo_dirs)
        for circ_idx, result in run_parallel_tasks(args.timeout,
                                                   len(circs),
                                                   get_all_circs_args,
                                                   cfg):

            circ = circs[circ_idx]
            ugly_name = circ[0]
            all_csvs = algo_csvs[ugly_name]
            if result is None:
                # Timeout
                re_csv = all_csvs
                _, lang_pretty, algo_pretty, params = circ
                re_csv.write(f'{lang_pretty}, {algo_pretty}, {params}, TIMEOUT\n')
            else:
                all_lines = result[0]
                all_csvs.writelines(all_lines)


if __name__ == '__main__':
    main()
