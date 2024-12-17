"""Utilities factored out of qre.py"""

import os.path
import qiskit
import qiskit.converters
from qiskit.circuit import Qubit, Clbit
from qiskit.dagcircuit import DAGInNode, DAGOutNode, DAGOpNode

# For custom transpiler
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes import BasisTranslator, HighLevelSynthesis, ElidePermutations, RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.passes import Depth, Size, FixedPoint, MinimumPoint
from qiskit.transpiler.passes.utils.gates_basis import GatesInBasis
from qiskit.transpiler.passes.synthesis.unitary_synthesis import UnitarySynthesis
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController
from qiskit.circuit.library.standard_gates.equivalence_library import StandardEquivalenceLibrary
from qiskit.transpiler.passes.optimization import (
    Optimize1qGatesDecomposition,
    CommutativeCancellation,
    Collect2qBlocks,
    ConsolidateBlocks,
    InverseCancellation,
)
from qiskit.circuit.library.standard_gates import (
    CXGate,
    ECRGate,
    CZGate,
    XGate,
    YGate,
    ZGate,
    TGate,
    TdgGate,
    SwapGate,
    SGate,
    SdgGate,
    HGate,
    CYGate,
    SXGate,
    SXdgGate,
)

def custom_transpile(circ, basis_gates, optimization_level):
    pm = PassManager()

    # "init" stage
    if optimization_level:
        pm.append(ElidePermutations())
        pm.append(RemoveDiagonalGatesBeforeMeasure())
        pm.append(
            InverseCancellation(
                [
                    CXGate(),
                    ECRGate(),
                    CZGate(),
                    CYGate(),
                    XGate(),
                    YGate(),
                    ZGate(),
                    HGate(),
                    SwapGate(),
                    (TGate(), TdgGate()),
                    (SGate(), SdgGate()),
                    (SXGate(), SXdgGate()),
                ]
            )
        )
        pm.append(CommutativeCancellation())
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())

    # "translate" stage
    equiv = StandardEquivalenceLibrary
    pm.append(HighLevelSynthesis(equivalence_library=equiv, basis_gates=basis_gates))
    pm.append(BasisTranslator(equiv, basis_gates))

    # "optimize" stage
    if optimization_level:
        _depth_check = [Depth(recurse=True), FixedPoint("depth")]
        _size_check = [Size(recurse=True), FixedPoint("size")]
        # Minimum point check for optimization level 3.
        _minimum_point_check = [
            Depth(recurse=True),
            Size(recurse=True),
            MinimumPoint(["depth", "size"], "optimization_loop"),
        ]

        # Steps for optimization level 3
        _opt = [
            # Collect2qBlocks(),
            # ConsolidateBlocks(basis_gates=basis_gates),
            # UnitarySynthesis(basis_gates),
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(),
        ]

        def _opt_control(property_set):
            return not property_set["optimization_loop_minimum_point"]

        unroll = BasisTranslator(equiv, basis_gates)

        # Build nested Flow controllers
        def _unroll_condition(property_set):
            return not property_set["all_gates_in_basis"]

        # Check if any gate is not in the basis, and if so, run unroll passes
        _unroll_if_out_of_basis = [
            GatesInBasis(basis_gates),
            ConditionalController(unroll, condition=_unroll_condition),
        ]

        pm.append(_minimum_point_check)

        opt_loop = _opt + _unroll_if_out_of_basis + _minimum_point_check
        pm.append(DoWhileController(opt_loop, do_while=_opt_control))
    return pm.run(circ)

def qiskit_transpile(circ, basis_gates, optimization_level):
    return qiskit.transpile(circ, basis_gates=basis_gates, optimization_level=optimization_level, routing_method='none', layout_method='trivial')

QSHARP_TO_QISKIT = {'m': 'measure'}
QISKIT_TO_QSHARP = dict((v, k) for k, v in QSHARP_TO_QISKIT.items())
blacklist = ['mresetz', 'estimate', 'qubit_allocate', 'qubit_release']

# Estimates the Cicrut
def estimate_circ(re, circ):
    dag = qiskit.converters.circuit_to_dag(circ)
    re_indices = {}
    for node in dag.topological_nodes():
        if isinstance(node, DAGInNode):
            if isinstance(node.wire, Clbit):
                # Who cares?
                pass
            elif isinstance(node.wire, Qubit):
                re_indices[node.wire] = re.qubit_allocate()
            else:
                raise NotImplementedError('unknown wire ' + repr(node.wire))
        elif isinstance(node, DAGOutNode):
            if isinstance(node.wire, Clbit):
                # Who cares?
                pass
            elif isinstance(node.wire, Qubit):
                re.qubit_release(re_indices[node.wire])
            else:
                raise NotImplementedError('unknown wire ' + repr(node.wire))
        elif isinstance(node, DAGOpNode):
            params = node.op.params
            arg_indices = [re_indices[qarg] for qarg in node.qargs]
            args = params + arg_indices

            op_name = QISKIT_TO_QSHARP.get(node.op.name, node.op.name)
            if not hasattr(re, op_name):
                raise NotImplementedError(f'unknown gate {op_name}')
            getattr(re, op_name)(*args)
        else:
            raise NotImplementedError('unknown dag node ' + repr(node))

def do_transpile(re, circ, opt_level, use_our_transpile):
    RE_GATES = [QSHARP_TO_QISKIT.get(attr, attr) for attr in dir(re) if not attr.startswith('_') and attr not in blacklist]
    transpile_func = custom_transpile if use_our_transpile else qiskit_transpile
    return transpile_func(circ, basis_gates=RE_GATES, optimization_level=opt_level)
