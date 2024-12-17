#!/usr/bin/env python3

import os
import os.path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import math
import argparse

DEFAULT_RESULTS_DIR = os.environ.get('QRE_RESULTS_DIR', os.path.join(os.path.dirname(__file__), 'results'))

qiskit = {}
qwerty = {}
qsharp = {}
quipper = {}

def initialize_specific_dictionary(dit):
    dit["bv"] = {}
    dit["dj_balanced"] = {}
    dit["dj_constant"] = {}
    dit["grover"] = {}
    dit["period"] = {}
    dit["simon"] = {}
    
    for key in dit:
        dit[key]["0"] = {}
        dit[key]["3"] = {}
        dit[key]["0"]["logical"] = {}
        dit[key]["0"]["physical"] = {}
        dit[key]["0"]["time"] = {}
        dit[key]["0"]["ccix"] = {}
        dit[key]["0"]["ccz"] = {}
        dit[key]["0"]["measure"] = {}
        dit[key]["0"]["rotation"] = {}
        dit[key]["0"]["depth"] = {}
        dit[key]["0"]["tgates"] = {}
        dit[key]["3"]["logical"] = {}
        dit[key]["3"]["physical"] = {}
        dit[key]["3"]["time"] = {}
        dit[key]["3"]["ccix"] = {}
        dit[key]["3"]["ccz"] = {}
        dit[key]["3"]["measure"] = {}
        dit[key]["3"]["rotation"] = {}
        dit[key]["3"]["depth"] = {}
        dit[key]["3"]["tgates"] = {}

def initialize_dictionary():
    initialize_specific_dictionary(qiskit)
    initialize_specific_dictionary(qwerty)
    initialize_specific_dictionary(qsharp)
    initialize_specific_dictionary(quipper)

def abbreviate(algo, type):
    abbrev = ""
    match algo:
        case "Bernstein-Vazirani":
            abbrev = "bv"
        case "Deutsch-Jozsa":
            match type:
                case "Balanced":
                    abbrev = "dj_balanced"
                case "Constant":
                    abbrev = "dj_constant"
                case _:
                    raise Exception("Error: Unknown Type for Deutsch-Jozsa")
        case "Grover":
            abbrev = "grover"
        case "Period":
            abbrev = "period"
        case "Simon":
            abbrev = "simon"
        case _:
            raise Exception("Error: Unknown Algorithm")
    return abbrev

def unabbreviate(algo):
    abbrev = ""
    match algo:
        case "bv":
            abbrev = "Bernstein-Vazirani"
        case "dj_balanced":
            abbrev = "Deutsch-Jozsa"
        case "dj_constant":
            abbrev = "Deutsch-Jozsa"
        case "grover":
            abbrev = "Grover"
        case "period":
            abbrev = "Period"
        case "simon":
            abbrev = "Simon"
        case _:
            raise Exception("Error: Unknown Algorithm")
    return abbrev

def parse_csv(results_dir, log):
    with open(os.path.join(results_dir, "results.csv"), "r") as file:
        for line in file:
            line = line.strip()
            linelst = [col.strip() for col in line.split(",")]

            if(linelst[0] == "Language"):
                continue

            abbrev = abbreviate(linelst[1], linelst[2])
            size = linelst[3]
            optimization = linelst[4]
            logical = linelst[5]
            physical = linelst[6]
            time = linelst[7]
            ccix = linelst[8]
            ccz = linelst[9]
            measure = linelst[10]
            rotation = linelst[11]
            depth = linelst[12]
            tgates = linelst[13]          

            if 'TIMEOUT' in linelst:
                raise ValueError('Cannot plot because the following result '
                                 'generation job timed out: ' +
                                 (', '.join(linelst[:linelst.index('TIMEOUT')])))

            match linelst[0]:
                case "Qwerty":
                    qwerty[abbrev][optimization]["logical"][size] = int(logical)
                    qwerty[abbrev][optimization]["physical"][size] = int(physical)
                    if(log):
                        qwerty[abbrev][optimization]["time"][size] = math.log(int(time))
                    else:
                        qwerty[abbrev][optimization]["time"][size] = int(time)
                    qwerty[abbrev][optimization]["ccix"][size] = int(ccix)
                    qwerty[abbrev][optimization]["ccz"][size] = int(ccz)
                    qwerty[abbrev][optimization]["measure"][size] = int(measure)
                    qwerty[abbrev][optimization]["rotation"][size] = int(rotation)
                    qwerty[abbrev][optimization]["depth"][size] = int(depth)
                    qwerty[abbrev][optimization]["tgates"][size] = int(tgates)

                case "Qiskit":
                    qiskit[abbrev][optimization]["logical"][size] = int(logical)
                    qiskit[abbrev][optimization]["physical"][size] = int(physical)
                    if(log):
                        qiskit[abbrev][optimization]["time"][size] = math.log(int(time))
                    else:
                        qiskit[abbrev][optimization]["time"][size] = int(time)
                    qiskit[abbrev][optimization]["ccix"][size] = int(ccix)
                    qiskit[abbrev][optimization]["ccz"][size] = int(ccz)
                    qiskit[abbrev][optimization]["measure"][size] = int(measure)
                    qiskit[abbrev][optimization]["rotation"][size] = int(rotation)
                    qiskit[abbrev][optimization]["depth"][size] = int(depth)
                    qiskit[abbrev][optimization]["tgates"][size] = int(tgates)

                case "Qsharp":
                    qsharp[abbrev][optimization]["logical"][size] = int(logical)
                    qsharp[abbrev][optimization]["physical"][size] = int(physical)
                    if(log):
                        qsharp[abbrev][optimization]["time"][size] = math.log(int(time))
                    else:
                        qsharp[abbrev][optimization]["time"][size] = int(time)
                    qsharp[abbrev][optimization]["ccix"][size] = int(ccix)
                    qsharp[abbrev][optimization]["ccz"][size] = int(ccz)
                    qsharp[abbrev][optimization]["measure"][size] = int(measure)
                    qsharp[abbrev][optimization]["rotation"][size] = int(rotation)
                    qsharp[abbrev][optimization]["depth"][size] = int(depth)
                    qsharp[abbrev][optimization]["tgates"][size] = int(tgates)

                case "Quipper":
                    quipper[abbrev][optimization]["logical"][size] = int(logical)
                    quipper[abbrev][optimization]["physical"][size] = int(physical)
                    if(log):
                        quipper[abbrev][optimization]["time"][size] = math.log(int(time))
                    else:
                        quipper[abbrev][optimization]["time"][size] = int(time)
                    quipper[abbrev][optimization]["ccix"][size] = int(ccix)
                    quipper[abbrev][optimization]["ccz"][size] = int(ccz)
                    quipper[abbrev][optimization]["measure"][size] = int(measure)
                    quipper[abbrev][optimization]["rotation"][size] = int(rotation)
                    quipper[abbrev][optimization]["depth"][size] = int(depth)
                    quipper[abbrev][optimization]["tgates"][size] = int(tgates)
                
                case _:
                    raise Exception("Error: Unknown Language")

def parse_args():
    """Parses and returns command line arguments."""
    parser = argparse.ArgumentParser(prog="graph",
                                    description="Graph data from the CSV File")
    parser.add_argument(
        "-r", "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Where to read result CSVs and write result graphs")
    parser.add_argument(
        "--ccix",
        action='store_true',
        default=False,
        help="Graph the CCIX count")
    parser.add_argument(
        "--ccz",
        action='store_true',
        default=False,
        help="Graph the CCZ count")
    parser.add_argument(
        "--measure",
        action='store_true',
        default=False,
        help="Graph the Measure count")
    parser.add_argument(
        "--rotation",
        action='store_true',
        default=False,
        help="Graph the Rotation count")
    parser.add_argument(
        "--depth",
        action='store_true',
        default=False,
        help="Graph the Rotation depth")
    parser.add_argument(
        "--tgates",
        action='store_true',
        default=False,
        help="Graph the T gate count")
    parser.add_argument(
        "--log_base_time",
        action='store_true',
        default=False,
        help="Convert time to log base for easier graph")
    return parser.parse_args()

# Remove first two data points. They clutter the plots
def chop(data):
    return data[2:]

def graph_bar(file, title, subytitle, sizes, qwerty, qiskit, quipper, qsharp, log):
    sizes, qwerty, qiskit, quipper, qsharp = (chop(x) for x in (sizes, qwerty, qiskit, quipper, qsharp))

    X_axis = np.arange(len(sizes)) 
    fig, ax = plt.subplots()
    opacity = 0.9
    ax.set_facecolor('white')

    plt.bar(X_axis - 0.3, qwerty, 0.2, alpha=opacity, label = "Asdf (Our Work)", color='#0072B2')
    plt.bar(X_axis - 0.1, qiskit, 0.2, alpha=opacity, label = "Qiskit", color='#009E73')
    plt.bar(X_axis + 0.1, quipper, 0.2, alpha=opacity, label = "Quipper", color='#F0E442')
    plt.bar(X_axis + 0.3, qsharp, 0.2, alpha=opacity, label = "Q#", color='#D55E00')
    
    plt.xlabel("Input size (bits)", fontsize= 14)
    plt.ylabel(subytitle, fontsize= 14)
    plt.title(title, fontsize=8)

    largest = int(max(max(qwerty),max(qiskit), max(quipper), max(qsharp)))
    smallest = int(min(min(qwerty), min(qiskit), min(quipper), min(qsharp)))
    #plt.ylim(top=largest+2)  # adjust the top leaving bottom unchanged
    #plt.ylim(bottom=smallest-2)

    plt.ticklabel_format(useOffset=False, style='plain')
    plt.xticks(X_axis, sizes, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    if log:
        plt.yscale('log')

    if 'Runtime' in subytitle:
        units = [('ns', 1), ('μs', 1e-3), ('ms', 1e-6), ('s', 1e-9)]
        idx = 0
        while largest > 9999:
            largest /= 1e3
            idx += 1
            if idx >= len(units): 
                idx = len(units) - 1
        mention_log = ', log scale' if log else ''
        plt.ylabel(f'Runtime ({units[idx][0]}{mention_log})', fontsize= 15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{int(y * units[idx][1])}'))
    elif 'Physical Kiloqubits' in subytitle:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{int(y * 1e-3)}'))

    if unabbreviate('bv') in title:
        plt.legend(fontsize=12)

    plt.tight_layout()
    plt.rcParams["figure.figsize"] = (4,3)
    #plt.savefig(file, dpi = 96 * 2)
    #plt.savefig(file, dpi = 8)
    plt.savefig(file)
    plt.close()

def graph_all(args):
    sizes = []
    for size in qwerty["bv"]["3"]["time"]:
        sizes.append(size)
    
    # Graph each component of the CSV file
    # Required: Time, Logical, Physical
    # Optional (based on flags): CCIX, CCZ, Measure, Rotation, Depth, T Gate

    qwerty_list = [0]*len(sizes)
    qiskit_list = [0]*len(sizes)
    quipper_list = [0]*len(sizes)
    qsharp_list = [0]*len(sizes)

    results_dir = args.results_dir

    # Time
    for algo in qwerty:
        for opt in qwerty[algo]:
            for index in range(len(sizes)):
                qwerty_list[index] = qwerty[algo][opt]["time"][sizes[index]]
                qiskit_list[index] = qiskit[algo][opt]["time"][sizes[index]]
                quipper_list[index] = quipper[algo][opt]["time"][sizes[index]]
                qsharp_list[index] = qsharp[algo][opt]["time"][sizes[index]]
            
            file = os.path.join(results_dir, f'{algo}_O{opt}_time.pdf')
            title = unabbreviate(algo) + ": Execution Time vs Input Size"
            ytitle = "Runtime (μs)"
            log_base_time = algo == 'grover'
            if(log_base_time):
                title = unabbreviate(algo) + ": Runtime vs Input Size"
                ytitle = "Runtime (μs, log scale)"
            graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, log_base_time)

    # Logical
    for algo in qwerty:
        for opt in qwerty[algo]:
            for index in range(len(sizes)):
                qwerty_list[index] = qwerty[algo][opt]["logical"][sizes[index]]
                qiskit_list[index] = qiskit[algo][opt]["logical"][sizes[index]]
                quipper_list[index] = quipper[algo][opt]["logical"][sizes[index]]
                qsharp_list[index] = qsharp[algo][opt]["logical"][sizes[index]]
            
            file = os.path.join(results_dir, f'{algo}_O{opt}_logical.pdf')
            title = unabbreviate(algo) + ": Logical Qubits vs Input Size"
            ytitle = "Logical Qubits"
            graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, False)
    
    # Physical
    for algo in qwerty:
        for opt in qwerty[algo]:
            for index in range(len(sizes)):
                qwerty_list[index] = qwerty[algo][opt]["physical"][sizes[index]]
                qiskit_list[index] = qiskit[algo][opt]["physical"][sizes[index]]
                quipper_list[index] = quipper[algo][opt]["physical"][sizes[index]]
                qsharp_list[index] = qsharp[algo][opt]["physical"][sizes[index]]
            
            file = os.path.join(results_dir, f'{algo}_O{opt}_physical.pdf')
            title = unabbreviate(algo) + ": Physical Kiloqubits vs Input Size"
            ytitle = "Physical Kiloqubits"
            graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, False)

    if(args.ccix):
        for algo in qwerty:
            for opt in qwerty[algo]:
                for index in range(len(sizes)):
                    qwerty_list[index] = qwerty[algo][opt]["ccix"][sizes[index]]
                    qiskit_list[index] = qiskit[algo][opt]["ccix"][sizes[index]]
                    quipper_list[index] = quipper[algo][opt]["ccix"][sizes[index]]
                    qsharp_list[index] = qsharp[algo][opt]["ccix"][sizes[index]]
            
                file = os.path.join(results_dir, f'{algo}_O{opt}_ccix.pdf')
                title = unabbreviate(algo) + ": CCIX vs Input Size"
                ytitle = "CCIX"
                graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, False)

    if(args.ccz):
        for algo in qwerty:
            for opt in qwerty[algo]:
                for index in range(len(sizes)):
                    qwerty_list[index] = qwerty[algo][opt]["ccz"][sizes[index]]
                    qiskit_list[index] = qiskit[algo][opt]["ccz"][sizes[index]]
                    quipper_list[index] = quipper[algo][opt]["ccz"][sizes[index]]
                    qsharp_list[index] = qsharp[algo][opt]["ccz"][sizes[index]]
            
                file = os.path.join(results_dir, f'{algo}_O{opt}_ccz.pdf')
                title = unabbreviate(algo) + ": CCZ vs Input Size"
                ytitle = "CCZ"
                graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, False)

    if(args.measure):
        for algo in qwerty:
            for opt in qwerty[algo]:
                for index in range(len(sizes)):
                    qwerty_list[index] = qwerty[algo][opt]["measure"][sizes[index]]
                    qiskit_list[index] = qiskit[algo][opt]["measure"][sizes[index]]
                    quipper_list[index] = quipper[algo][opt]["measure"][sizes[index]]
                    qsharp_list[index] = qsharp[algo][opt]["measure"][sizes[index]]
            
                file = os.path.join(results_dir, f'{algo}_O{opt}_measure.pdf')
                title = unabbreviate(algo) + ": Measurement Count vs Input Size"
                ytitle = "Measurement Count"
                graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, False)

    if(args.rotation):
        for algo in qwerty:
            for opt in qwerty[algo]:
                for index in range(len(sizes)):
                    qwerty_list[index] = qwerty[algo][opt]["rotation"][sizes[index]]
                    qiskit_list[index] = qiskit[algo][opt]["rotation"][sizes[index]]
                    quipper_list[index] = quipper[algo][opt]["rotation"][sizes[index]]
                    qsharp_list[index] = qsharp[algo][opt]["rotation"][sizes[index]]
            
                file = os.path.join(results_dir, f'{algo}_O{opt}_rotation.pdf')
                title = unabbreviate(algo) + ": Rotation Count vs Input Size"
                ytitle = "Rotation Count"
                graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, False)

    if(args.depth):
        for algo in qwerty:
            for opt in qwerty[algo]:
                for index in range(len(sizes)):
                    qwerty_list[index] = qwerty[algo][opt]["depth"][sizes[index]]
                    qiskit_list[index] = qiskit[algo][opt]["depth"][sizes[index]]
                    quipper_list[index] = quipper[algo][opt]["depth"][sizes[index]]
                    qsharp_list[index] = qsharp[algo][opt]["depth"][sizes[index]]
            
                file = os.path.join(results_dir, f'{algo}_O{opt}_depth.pdf')
                title = unabbreviate(algo) + ": Rotation Depth vs Input Size"
                ytitle = "Rotation Depth"
                graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, False)
        
    if(args.tgates):
        for algo in qwerty:
            for opt in qwerty[algo]:
                for index in range(len(sizes)):
                    qwerty_list[index] = qwerty[algo][opt]["tgates"][sizes[index]]
                    qiskit_list[index] = qiskit[algo][opt]["tgates"][sizes[index]]
                    quipper_list[index] = quipper[algo][opt]["tgates"][sizes[index]]
                    qsharp_list[index] = qsharp[algo][opt]["tgates"][sizes[index]]
            
                file = os.path.join(results_dir, f'{algo}_O{opt}_tgates.pdf')
                title = unabbreviate(algo) + ": T Gate Count vs Input Size"
                ytitle = "T Gate Count"
                graph_bar(file, title, ytitle, sizes, qwerty_list, qiskit_list, quipper_list, qsharp_list, False)

def main():
    args = parse_args()
    initialize_dictionary()
    parse_csv(args.results_dir, False)
    graph_all(args)

if __name__ == '__main__':
    main()
