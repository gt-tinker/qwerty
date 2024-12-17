import System.Environment
import Control.Monad

-- Import Quipper library
import Quipper
import Quipper.Utils.Auxiliary

-- Define Oracle data type
data Oracle = Oracle {
    qubit_num :: Int,
    function :: ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
}

-- Simon's circuit function
simon_circuit :: Oracle -> Circ [Bit]
simon_circuit oracle = do
    -- Initialize qubits
    top_qubits <- qinit (replicate (qubit_num oracle) False)
    bottom_qubits <- qinit (replicate (qubit_num oracle) False)

    -- Apply Hadamard gate to top qubits, producing |+>^n
    top_qubits <- mapUnary hadamard top_qubits

    -- Call the oracle function
    (top_qubits, bottom_qubits) <- function oracle (top_qubits, bottom_qubits)

    -- Apply Hadamard again to the top qubits
    top_qubits <- mapUnary hadamard top_qubits

    -- Measure the bottom qubits and discard them
    bottom_bits <- measure bottom_qubits
    cdiscard bottom_bits

    -- Measure the top qubits
    top_bits <- measure top_qubits
    return top_bits

-- Synthesized oracle

-- These four are taken from Algorithms/BF/Hex.hs inside the Quipper source
-- code. It is unclear to me why it is necessary that programmers write these
-- manually

-- | 'Int' is not changed along the conversion.
template_integer :: Int -> Circ Int
template_integer x = return x

-- | A hand-lifted version of the 'take' function, specialized to lists of qubits.
template_take :: Circ (Int -> Circ ([Qubit] -> Circ [Qubit]))
template_take = return $ \n -> return $ \qs -> return (take n qs)

-- | A hand-lifted version of the 'drop' function, specialized to lists of qubits.
template_drop :: Circ (Int -> Circ ([Qubit] -> Circ [Qubit]))
template_drop = return $ \n -> return $ \qs -> return (drop n qs)

-- | A hand-lifted function to get the 'length' of a list.
template_length :: Circ ([a] -> Circ Int)
template_length = return $ \as -> return $ length as

-- Now this one I improvised at least based on their template_symb_plus_.
-- Lord have mercy
template_quot :: Circ (Int -> Circ (Int -> Circ Int))
template_quot = return $ \x -> return $ \y -> return (quot x y)

-- Now, back to the code...

build_circuit
oracle_tail_helper :: Bool -> [Bool] -> [Bool]
oracle_tail_helper bx (by:bs) = (bool_xor bx by) : (oracle_tail_helper bx bs)
oracle_tail_helper bx [] = []

build_circuit
oracle_tail :: [Bool] -> [Bool]
oracle_tail (bx:(by:bs)) = False : (oracle_tail_helper bx (by:bs))
oracle_tail [] = []

build_circuit
oracle :: [Bool] -> [Bool]
oracle bs =
    let k = quot (length bs) 2 in
    (take k bs) ++ (oracle_tail (drop k bs))

synth_oracle :: ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
synth_oracle = classical_to_reversible (unpack template_oracle)

-- Handwritten oracle

handwritten_oracle :: ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
handwritten_oracle (x, y) = do
    (x_head, x_pivot, x_tail) <- split x
    (y_head, y_pivot, y_tail) <- split y

    zipWithM_ (\xi yi -> qnot_at yi `controlled` xi) x_head y_head
    zipWithM_ (\xi yi -> qnot_at yi `controlled` xi) x_tail y_tail
    mapM_ (\yi -> qnot_at yi `controlled` x_pivot) y_tail

    return (x, y)
    where
        split :: [Qubit] -> Circ ([Qubit], Qubit, [Qubit])
        split x = do
            let n = length x
            let k = quot n 2
            return ((take k x), (head (drop k x)), (tail (drop k x)))

-- Main function
main :: IO ()
main = do
    args <- getArgs
    let oracle = get_oracle (parse_args args)
    print_simple ASCII (simon_circuit oracle)

parse_args :: [String] -> (String, Int)
parse_args [impl, n_qubits_str] = (impl, read n_qubits_str)
parse_args _ = error "usage: ./simon synth|hand <n_qubits>"

get_oracle :: (String, Int) -> Oracle
get_oracle ("synth", n) = Oracle n synth_oracle
get_oracle ("hand", n) = Oracle n handwritten_oracle
get_oracle _ = error "oracle undefined"
