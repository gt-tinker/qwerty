import System.Environment

-- Import Quipper library
import Quipper

-- Define the Oracle data type
data Oracle = Oracle {
    qubit_num :: Int,
    function :: ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
}

-- Phase inversion function
phase_inversion :: (([Qubit], Qubit) -> Circ ([Qubit], Qubit)) -> ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
phase_inversion oracle (top_qubits, bottom_qubit) = do
    -- Apply the oracle function to the qubits
    oracle (top_qubits, bottom_qubit)
    return (top_qubits, bottom_qubit)

-- Inversion about mean function
inversion_about_mean :: ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
inversion_about_mean (top_qubits, bottom_qubit) = do
    -- Apply X gate to top qubits
    mapUnary gate_X top_qubits

    -- Separate target and control qubits
    let target_qubit = last top_qubits
    let controlled_qubits = init top_qubits

    -- Apply Hadamard and conditional phase shift operations
    hadamard_at target_qubit
    qnot_at target_qubit `controlled` controlled_qubits
    hadamard_at target_qubit

    -- Apply X gate again to top qubits
    mapUnary gate_X top_qubits
    return (top_qubits, bottom_qubit)

-- Grover search circuit
grover_search_circuit :: Oracle -> Int -> Circ ([Bit])
grover_search_circuit oracle n_iter = do
    -- Initialize qubits
    let n = qubit_num oracle
    -- let n_iter = floor (sqrt (2 ** fromIntegral n))
    top <- qinit (replicate n False)
    bottom <- qinit True

    -- Apply Hadamard gate to all qubits
    mapUnary hadamard top
    hadamard_at bottom

    label bottom ("Iterations: " ++ (show n_iter))

    -- Grover's iteration loop
    for 1 n_iter 1 $ \_ -> do
        -- Phase inversion
        (top, bottom) <- phase_inversion (function oracle) (top, bottom)

        -- Inversion about mean
        (top, bottom) <- inversion_about_mean (top, bottom)
        return ()

    -- Measure the qubits and return result
    hadamard_at bottom
    (top, bottom) <- measure (top, bottom)
    cdiscard bottom
    return top

-- Synthesized oracle

build_circuit
oracle :: [Bool] -> Bool
oracle [b] = b
oracle (b:bs) = b && (oracle bs)
oracle _ = error "Nonzero number of qubits needed"

synth_oracle = classical_to_reversible (unpack template_oracle)

-- Handwritten oracle

handwritten_oracle :: ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
handwritten_oracle (controls, target) = do
    qnot_at target `controlled` controls
    return (controls, target)

-- Main function
main :: IO ()
main = do
    args <- getArgs
    let (impl, n_qubits, n_iter) = parse_args args
    let oracle = get_oracle impl n_qubits
    print_simple ASCII (grover_search_circuit oracle n_iter)

parse_args :: [String] -> (String, Int, Int)
parse_args [impl, n_qubits_str, n_iter_str] = (impl, read n_qubits_str, read n_iter_str)
parse_args _ = error "usage: ./grover synth|hand <n_qubits> <n_iter>"

get_oracle :: String -> Int -> Oracle
get_oracle "synth" n = Oracle n synth_oracle
get_oracle "hand" n = Oracle n handwritten_oracle
get_oracle _ _ = error "oracle undefined"
