import System.Environment

-- Import Quipper library
import Quipper
import Quipper.Utils.Auxiliary

-- Define Oracle data type
data Oracle = Oracle {
    qubit_num :: Int,
    function :: ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
}

-- Deutsch-Jozsa circuit function
deutsch_jozsa_circuit :: Oracle -> Circ [Bit]
deutsch_jozsa_circuit oracle = do
    -- Initialize qubits
    top_qubits <- qinit (replicate (qubit_num oracle) False)
    bottom_qubit <- qinit True
    label (top_qubits, bottom_qubit) ("|0>","|1>")

    -- Apply Hadamard gate to all qubits
    mapUnary hadamard top_qubits
    hadamard_at bottom_qubit

    -- Call the oracle function
    (top_qubits, bottom_qubit) <- function oracle (top_qubits, bottom_qubit)

    -- Apply Hadamard again to the top qubits
    top_qubits <- mapUnary hadamard top_qubits

    -- Measure the bottom qubit and discard it
    bottom_bit <- measure bottom_qubit
    cdiscard bottom_bit

    -- Measure the top qubits
    top_bits <- measure top_qubits
    return top_bits

-- Constant oracle

build_circuit
constant_oracle :: [Bool] -> Bool
constant_oracle _ = True

synth_constant_oracle = classical_to_reversible (unpack template_constant_oracle)

handwritten_constant_oracle :: ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
handwritten_constant_oracle (x, out) = do
    out <- qnot out
    return (x, out)

-- Balanced oracle

build_circuit
balanced_oracle :: [Bool] -> Bool
balanced_oracle (b:bs) = bool_xor b (balanced_oracle bs)
balanced_oracle [] = False

synth_balanced_oracle = classical_to_reversible (unpack template_balanced_oracle)

handwritten_balanced_oracle :: ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
handwritten_balanced_oracle (x:xs, out) = do
    -- Apply controlled-not gates
    qnot_at out `controlled` x
    (xs, out) <- handwritten_balanced_oracle (xs, out)
    return (x:xs, out)
handwritten_balanced_oracle ([], out) = do
    return ([], out)

-- Main function
main :: IO ()
main = do
    args <- getArgs
    let oracle = get_oracle (parse_args args)
    print_simple ASCII (deutsch_jozsa_circuit oracle)

parse_args :: [String] -> (String, String, Int)
parse_args [kind, impl, n_qubits_str] = (kind, impl, read n_qubits_str)
parse_args _ = error "usage: ./dj constant|balanced synth|hand <n_qubits>"

get_oracle :: (String, String, Int) -> Oracle
get_oracle ("balanced", "synth", n) = Oracle n synth_balanced_oracle
get_oracle ("balanced", "hand", n) = Oracle n handwritten_balanced_oracle
get_oracle ("constant", "synth", n) = Oracle n synth_constant_oracle
get_oracle ("constant", "hand", n) = Oracle n handwritten_constant_oracle
get_oracle _ = error "oracle undefined"
