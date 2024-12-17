#!/usr/bin/env python3

import os
import json
import time
import queue
import argparse
import contextlib
import multiprocessing
import multiprocessing.connection
from dataclasses import dataclass
from collections import OrderedDict

from qwerty import *
import qsharp
import qiskit
import qiskit.qasm2
import qiskit.qasm3
from qiskit_aer import AerSimulator

from estimate_utils import estimate_circ, do_transpile

from benchmarks.dj import dj_qwerty, dj_qiskit, dj_qsharp, dj_quipper
from benchmarks.bv import bv_qwerty, bv_qsharp, bv_qiskit, bv_quipper
from benchmarks.grover import grover_qsharp, grover_qwerty, grover_qiskit, grover_quipper
from benchmarks.simon import simon_qsharp, simon_qwerty, simon_qiskit, simon_quipper
from benchmarks.period import period_qsharp, period_qwerty, period_qiskit, period_quipper

def parameters_split(s):
    x = s.replace(" ", "").split(",")
    return x

# ------------------------------------Running Algorithms----------------------------------------
def dj_balanced_circs(problem_size):
    return [
        ('dj_balanced', 'Qwerty', 'Deutsch-Jozsa', 'Balanced', lambda: dj_qwerty.get_circuit(dj_qwerty.balanced[[problem_size]]), problem_size),
        ('dj_balanced', 'Qsharp', 'Deutsch-Jozsa', 'Balanced', lambda: dj_qsharp.get_circuit(problem_size, 1), problem_size),
        ('dj_balanced', 'Quipper', 'Deutsch-Jozsa', 'Balanced', lambda: dj_quipper.get_circuit('balanced', problem_size), problem_size),
        ('dj_balanced', 'Qiskit', 'Deutsch-Jozsa', 'Balanced', lambda: dj_qiskit.get_circuit(dj_qiskit.balanced(problem_size)), problem_size),
    ]
def dj_constant_circs(problem_size):
    return [
        ('dj_constant', 'Qwerty', 'Deutsch-Jozsa', 'Constant', lambda: dj_qwerty.get_circuit(dj_qwerty.constant[[problem_size]]), problem_size),
        ('dj_constant', 'Qsharp', 'Deutsch-Jozsa', 'Constant', lambda: dj_qsharp.get_circuit(problem_size, 0), problem_size),
        ('dj_constant', 'Quipper', 'Deutsch-Jozsa', 'Constant', lambda: dj_quipper.get_circuit('constant', problem_size), problem_size),
        ('dj_constant', 'Qiskit', 'Deutsch-Jozsa', 'Constant', lambda: dj_qiskit.get_circuit(dj_qiskit.constant(problem_size)), problem_size),
    ]
def bv_circs(problem_size):
    secret_string = '10'*(problem_size//2)
    return [
        ('bv', 'Qwerty', 'Bernstein-Vazirani', 'Alternating', lambda: bv_qwerty.get_circuit(bv_qwerty.get_black_box(bit.from_str(secret_string))), problem_size),
        ('bv', 'Qsharp', 'Bernstein-Vazirani', 'Alternating', lambda: bv_qsharp.get_circuit(problem_size, secret_string), problem_size),
        ('bv', 'Qiskit', 'Bernstein-Vazirani', 'Alternating', lambda: bv_qiskit.get_circuit(problem_size, secret_string), problem_size),
        ('bv', 'Quipper', 'Bernstein-Vazirani', 'Alternating', lambda: bv_quipper.get_circuit(secret_string), problem_size),
    ]
def grover_circs(problem_size):
    # 8 qubits has 12 iterations. Let's max out there to make this run
    # in reasonable time
    saturated_problem_size = min(problem_size, 8)
    # All ones by default
    n_iter = grover_qwerty.get_n_iter(saturated_problem_size, n_answers=1)
    return [
        ('grover', 'Qwerty', 'Grover', 'All Ones', lambda: grover_qwerty.get_circuit(problem_size, n_iter), problem_size),
        ('grover', 'Qsharp', 'Grover', 'All Ones', lambda: grover_qsharp.get_circuit(problem_size, n_iter), problem_size),
        ('grover', 'Qiskit', 'Grover', 'All Ones', lambda: grover_qiskit.get_circuit(problem_size, n_iter), problem_size),
        ('grover', 'Quipper', 'Grover', 'All Ones', lambda: grover_quipper.get_circuit(problem_size, n_iter), problem_size),
    ]
def simon_circs(problem_size):
    return [
        ('simon', 'Qwerty', 'Simon', 'Secret Sauce', lambda: simon_qwerty.get_circuit(problem_size), problem_size),
        ('simon', 'Qsharp', 'Simon', 'Secret Sauce', lambda: simon_qsharp.get_circuit(problem_size), problem_size),
        ('simon', 'Qiskit', 'Simon', 'Secret Sauce', lambda: simon_qiskit.get_circuit(problem_size), problem_size),
        ('simon', 'Quipper', 'Simon', 'Secret Sauce', lambda: simon_quipper.get_circuit(problem_size), problem_size),
    ]
def period_circs(problem_size):
    n_bits_in = problem_size
    n_bits_out = n_bits_in-1
    n_mask_bits = problem_size//2
    return [
        ('period', 'Qwerty', 'Period', 'Masking', lambda: period_qwerty.get_circuit(n_bits_in, n_bits_out, n_mask_bits), problem_size),
        ('period', 'Qsharp', 'Period', 'Masking', lambda: period_qsharp.get_circuit(n_bits_in, n_bits_out, n_mask_bits), problem_size),
        ('period', 'Qiskit', 'Period', 'Masking', lambda: period_qiskit.get_circuit(n_bits_in, n_bits_out, n_mask_bits), problem_size),
        ('period', 'Quipper', 'Period', 'Masking', lambda: period_quipper.get_circuit(n_bits_in, n_bits_out, n_mask_bits), problem_size),
    ]

def get_all_circs(arg_algo, arg_problem_sizes):
    if arg_problem_sizes:
        problem_sizes = arg_problem_sizes
    else:
        problem_sizes = [2**x for x in range(2, 8)]
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
        for problem_size in problem_sizes:
            circs.extend(algo_circ_funcs[arg_algo](problem_size))
    elif arg_algo == 'all':
        for problem_size in problem_sizes:
            for circ_func in algo_circ_funcs.values():
                circs.extend(circ_func(problem_size))
    else:
        raise ValueError('Invalid algorithm name')

    circs_with_opt = [circ + (opt,) for opt in (True, False)
                                    for circ in circs]
    return circs_with_opt

# --------------------------------------------------------------------------------------------------

def get_opt_level(do_opt):
    return 3 if do_opt else 0

def evaluate_circ(circ, qasm_dir, circs_dir, re_params='', do_simulation=False, use_our_transpile=False):
    # Initialize Resource Estimator and Circuit
    algo_ugly, lang_pretty, algo_pretty, params, gen_func, problem_size, do_opt = circ
    opt_level = get_opt_level(do_opt)
    circ, circ_qasm = gen_func()
    re = qsharp.ResourceEstimator(re_params)
    print(f"{algo_pretty} in {lang_pretty} (params: {params}) [Problem Size: {problem_size}]")
    print(f"\tTranspiling")
    opt_circ = do_transpile(re, circ, opt_level, use_our_transpile)
    print(f"\tTranspiling Complete")

    # Run Estimator over Circuit
    print(f"\tEvaluating")
    estimate_circ(re, opt_circ)
    result = json.loads(re.estimate())
    print(f"\tEvaluation Complete")

    print(f"\tDumping Results")

    # Dump Qasm Result into file
    with open(os.path.join(qasm_dir, f'{lang_pretty}_{problem_size}.qasm'), 'w') as raw_qasm_out:
        header = f"_________{lang_pretty}\'s Qasm [Problem Size: {problem_size}]_________"
        raw_qasm_out.write(f'{header}\n{circ_qasm}\n')

    if problem_size <= 8:
        # Dump Quantum Circuit Result into file
        with open(os.path.join(circs_dir, f'{lang_pretty}_{problem_size}.circ'), 'w') as raw_circ_out:
            header = f"_________{lang_pretty}\'s Circuit [Problem Size: {problem_size}]_________"
            raw_circ_out.write(f'{header}\n{opt_circ}\n')

    # Run Simulator and dump result into file
    if do_simulation:
        aer_sim = AerSimulator(method='statevector')
        opt_circ.remove_final_measurements(True)
        opt_circ.save_statevector()
        output_state = aer_sim.run(opt_circ).result().get_statevector(opt_circ)
        header = f"_________{lang_pretty}\'s Circuit Result_________"
        simulator_res = [f'{header}\n{output_state.data.round(3)}\n']
    else:
        simulator_res = []

    # Dump estimator results into csv
    re_res = []
    physicalQubits = result[0]['physicalCounts']['physicalQubits']
    runtime = result[0]['physicalCounts']['runtime']
    logicalQubits = result[0]['logicalCounts']['numQubits']
    ccixCount = result[0]['logicalCounts']['ccixCount']
    cczCount = result[0]['logicalCounts']['cczCount']
    measurementCount = result[0]['logicalCounts']['measurementCount']
    rotationCount = result[0]['logicalCounts']['rotationCount']
    rotationDepth = result[0]['logicalCounts']['rotationDepth']
    tCount = result[0]['logicalCounts']['tCount']
    re_res.append(f'{lang_pretty}, {algo_pretty}, {params}, {problem_size}, {opt_level}, {logicalQubits}, {physicalQubits}, {runtime}, {ccixCount}, {cczCount}, {measurementCount}, {rotationCount}, {rotationDepth}, {tCount}\n')

    return re_res, simulator_res

@dataclass
class AlgoDirs:
    algo_dir: str
    raw_circs_dir: str
    raw_qasm_dir: str
    opt_circs_dir: str
    opt_qasm_dir: str

DEFAULT_RESULTS_DIR = os.environ.get('QRE_RESULTS_DIR', os.path.join(os.path.dirname(__file__), 'results'))

def make_algo_dir(results_dir, algo_ugly):
    # Create Algo Folder in results directory
    algo_folder = os.path.join(results_dir, algo_ugly)
    if not os.path.exists(algo_folder):
        os.makedirs(algo_folder)
    # Raw Folder
    raw_folder = os.path.join(algo_folder, "raw/")
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)
    raw_circs_folder = os.path.join(raw_folder, "circs/")
    raw_qasm_folder = os.path.join(raw_folder, "qasm/")
    if not os.path.exists(raw_circs_folder):
        os.makedirs(raw_circs_folder)
    if not os.path.exists(raw_qasm_folder):
        os.makedirs(raw_qasm_folder)
    # Optimized Folder
    opt_folder = os.path.join(algo_folder, "opt/")
    if not os.path.exists(opt_folder):
        os.makedirs(opt_folder)
    opt_circs_folder = os.path.join(opt_folder, "circs/")
    opt_qasm_folder = os.path.join(opt_folder, "qasm/")
    if not os.path.exists(opt_circs_folder):
        os.makedirs(opt_circs_folder)
    if not os.path.exists(opt_qasm_folder):
        os.makedirs(opt_qasm_folder)

    return AlgoDirs(algo_folder,
                    raw_circs_folder,
                    raw_qasm_folder,
                    opt_circs_folder,
                    opt_qasm_folder)

@dataclass
class Config:
    all_algo_dirs: dict[str, AlgoDirs]
    re_params: str
    do_simulation: bool
    use_our_transpile: bool

def get_all_tasks(*args, **kwargs):
    return get_all_circs(*args, **kwargs)

def run_task(circ, cfg):
    ugly_name = circ[0]
    algo_dirs = cfg.all_algo_dirs[ugly_name]
    do_opt = circ[-1]

    if do_opt:
        do_simulation = False
        qasm_dir = algo_dirs.opt_qasm_dir
        circs_dir = algo_dirs.opt_circs_dir
    else:
        qasm_dir = algo_dirs.raw_qasm_dir
        circs_dir = algo_dirs.raw_circs_dir

    return evaluate_circ(circ, qasm_dir, circs_dir, cfg.re_params, cfg.do_simulation, cfg.use_our_transpile)

def run_process(out_queue, task_idx, all_tasks_args, cfg):
    tasks = get_all_tasks(*all_tasks_args)
    task = tasks[task_idx]
    ret = run_task(task, cfg)
    out_queue.put_nowait(ret)

@dataclass
class RunningProcess:
    proc: multiprocessing.Process
    queue: multiprocessing.Queue
    circ_idx: int
    start_sec: float

def run_parallel_tasks(timeout_sec, n_tasks, all_tasks_args, get_all_tasks, cfg):
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
                print('!!!!!!! TIMEOUT')
                proc.queue.close()
                proc.proc.terminate()
                proc.proc.join(timeout=2.0)
                if proc.proc.exitcode is None:
                    raise ValueError('Join is blocking after we sent '
                                     'SIGTERM, how?')
                proc.proc.close()
                procs[proc_idx] = None

                yield (proc.circ_idx, None)

def fake_run_parallel_tasks(_timeout_sec, n_tasks, all_tasks_args, get_all_tasks, cfg):
    tasks = get_all_tasks(*all_tasks_args)
    for task_idx in range(n_tasks):
        task = tasks[task_idx]
        ret = run_task(task, cfg)
        yield (task_idx, ret)

def main():
    parser = argparse.ArgumentParser(description='Resource Estimator')
    parser.add_argument(
        '-r',
        '--results-dir',
        help='Where to put results',
        default=None
    )
    parser.add_argument(
        '-s',
        '--simulator',
        help= 'Run Simulator',
        action= 'store_true'
    )
    parser.add_argument(
        '-a',
        '--algorithm',
        type = str,
        default = "all",
        help='type the name of the algorithm you want to test over.'
    )
    parser.add_argument(
        '-p',
        '--parameters',
        type=parameters_split,
        help='parameters for the function'
    )
    parser.add_argument(
        '-n',
        '--problem-sizes',
        type=int,
        nargs='+',
        default=None,
        help='The size of our input to our algorithm'
    )
    parser.add_argument(
        '-t',
        '--timeout',
        type=float,
        default=4*60*60,
        help='Per-job timeout in seconds (default: 4 hours)'
    )
    parser.add_argument(
        '-x',
        '--single-threaded',
        action='store_true',
        help='Run in a single thread (for debugging)'
    )
    parser.add_argument(
        '-u',
        '--use-vanilla-transpile',
        action='store_true',
        help='Use qiskit transpile() for transpiling instead of our lightly modified version'
    )
    args = parser.parse_args()

    # Tell Q# to use the base profile. This way it will use the versions of
    # functions that are compatible with .circuit()
    qsharp.init(target_profile=qsharp.TargetProfile.Base)

    # Create results directory
    results_dir = args.results_dir or DEFAULT_RESULTS_DIR
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    re_params = ''
    get_all_circs_args = (args.algorithm.lower(), args.problem_sizes)
    circs = get_all_circs(*get_all_circs_args)

    ugly_names = {c[0] for c in circs}
    algo_dirs = {ugly_name: make_algo_dir(results_dir, ugly_name) for ugly_name in ugly_names}

    with contextlib.ExitStack() as exit_stack:
        algo_csvs = {ugly_name: (exit_stack.enter_context(
                                     open(os.path.join(algo_dirs[ugly_name].algo_dir,
                                                       'resource_estimation.csv'), 'w')),
                                 exit_stack.enter_context(
                                     open(os.path.join(algo_dirs[ugly_name].algo_dir,
                                                       'simulator_result.out'), 'w')))
                     for ugly_name in ugly_names}

        for csvs in algo_csvs.values():
            re_csv, _ = csvs
            re_csv.write(f'Language, Algorithm, Type, Problem Size, Optimization Level, Logical Qubits, Physical Qubits, Runtime, ccixCount, cczCount, measurementCount, rotationCount, rotationDepth, tCount\n')

        cfg = Config(algo_dirs, re_params, args.simulator, not args.use_vanilla_transpile)
        runner = fake_run_parallel_tasks if args.single_threaded \
                 else run_parallel_tasks
        for circ_idx, result in runner(args.timeout,
                                       len(circs),
                                       get_all_circs_args,
                                       get_all_circs, cfg):

            circ = circs[circ_idx]
            ugly_name = circ[0]
            all_csvs = algo_csvs[ugly_name]
            if result is None:
                # Timeout
                re_csv, _ = all_csvs
                _, lang_pretty, algo_pretty, params, _, problem_size, do_opt = circ
                opt_level = get_opt_level(do_opt)
                re_csv.write(f'{lang_pretty}, {algo_pretty}, {params}, {problem_size}, {opt_level}, TIMEOUT, TIMEOUT, TIMEOUT, TIMEOUT, TIMEOUT, TIMEOUT, TIMEOUT, TIMEOUT, TIMEOUT\n')
            else:
                all_lines = result
                for lines, fp in zip(all_lines, all_csvs):
                    fp.writelines(lines)

if __name__ == '__main__':
    main()
