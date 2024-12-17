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
bernstein_vazirani_circuit :: Oracle -> Circ [Bit]
bernstein_vazirani_circuit oracle = do
    -- Initialize qubits
    top_qubits <- qinit (replicate (qubit_num oracle) False)
    bottom_qubit <- qinit True
    label (top_qubits, bottom_qubit) ("|0>","|1>")

    -- Apply Hadamard gate to all qubits
    top_qubits <- mapUnary hadamard top_qubits
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

-- Synthesized oracle

build_circuit
oracle :: [BoolParam] -> [Bool] -> Bool
oracle (s:ss) (b:bs) =
    -- Can't do BoolParam && Bool, so we resort to this ugly thing
    case s of
        PTrue -> bool_xor b (oracle ss bs)
        PFalse -> oracle ss bs
oracle [] [] = False
oracle _ _ = error "Mismatch in list sizes"

bool_to_param :: Bool -> BoolParam
bool_to_param True = PTrue
bool_to_param False = PFalse

synth_oracle :: [Bool] -> ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
synth_oracle secret_string = classical_to_reversible ((unpack template_oracle) (map bool_to_param secret_string))

-- Handwritten oracle

handwritten_oracle :: [Bool] -> ([Qubit], Qubit) -> Circ ([Qubit], Qubit)
handwritten_oracle (s:ss) ((x:xs), out) = do
    -- Apply controlled-not gates
    qnot_at out `controlled` (x, s)
    (xs, out) <- handwritten_oracle ss (xs, out)
    return ((x:xs), out)
handwritten_oracle [] ([], out) = do
    return ([], out)
handwritten_oracle _ _ = error "Mismatch in list sizes"

-- Main function
main :: IO ()
main = do
    args <- getArgs
    let oracle = get_oracle (parse_args args)
    print_simple ASCII (bernstein_vazirani_circuit oracle)

parse_args :: [String] -> (String, Int, [Bool])
parse_args [impl, secret_str] = (impl, (length secret_str), (parse_secret_str secret_str))
parse_args _ = error "usage: ./bv synth|hand <secret_string>"

parse_secret_str :: String -> [Bool]
parse_secret_str s = [case c of
                      '0' -> False;
                      '1' -> True;
                       _  -> error "invalid bit"
                      | c <- s]

get_oracle :: (String, Int, [Bool]) -> Oracle
get_oracle ("synth", n, secret_string) = Oracle n (synth_oracle secret_string)
get_oracle ("hand", n, secret_string) = Oracle n (handwritten_oracle secret_string)
get_oracle _ = error "oracle undefined"
