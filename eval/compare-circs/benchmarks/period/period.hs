import System.Environment
import Control.Monad

-- Import Quipper library
import Quipper
import Quipper.Libraries.QFT
import Quipper.Utils.Auxiliary

-- Define Oracle data type
data Oracle = Oracle {
    n_bits_in :: Int,
    n_bits_out :: Int,
    function :: ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
}

-- Period finding circuit function
period_circuit :: Oracle -> Circ [Bit]
period_circuit oracle = do
    -- Initialize qubits
    top_qubits <- qinit (replicate (n_bits_in oracle) False)
    bottom_qubits <- qinit (replicate (n_bits_out oracle) False)

    -- Apply Hadamard gate to top qubits, producing |+>^n
    top_qubits <- mapUnary hadamard top_qubits

    -- Call the oracle function
    (top_qubits, bottom_qubits) <- function oracle (top_qubits, bottom_qubits)

    -- Apply Hadamard again to the top qubits
    top_qubits <- (reverse_generic_endo qft_big_endian) top_qubits

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

-- | A hand-lifted version of the 'drop' function, specialized to lists of qubits.
template_drop :: Circ (Int -> Circ ([Qubit] -> Circ [Qubit]))
template_drop = return $ \n -> return $ \qs -> return (drop n qs)

-- | A hand-lifted function to get the 'length' of a list.
template_length :: Circ ([a] -> Circ Int)
template_length = return $ \as -> return $ length as

-- | A hand-lifted version of the '-' function, specialized to 'Int'.
template_symb_minus_ :: Circ (Int -> Circ (Int -> Circ Int))
template_symb_minus_ = return $ \x -> return $ \y -> return (x - y)

-- Now, back to the code...

build_circuit
oracle_ones :: Int -> [Bool]
oracle_ones 0 = []
oracle_ones n = True : (oracle_ones (n-1))

build_circuit
oracle :: (Int, Int) -> [Bool] -> [Bool]
oracle (n_bits_out, n_mask_bits) bs =
    let n_bits_in = length bs in
    (oracle_ones (n_bits_out - n_mask_bits)) ++ (drop (n_bits_in - n_mask_bits) bs)

synth_oracle :: (Int, Int) -> ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
synth_oracle cfg = classical_to_reversible ((unpack template_oracle) cfg)

-- Handwritten oracle

handwritten_oracle :: Int -> ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
handwritten_oracle n_mask_bits (x, y) = do
    let n_bits_in = length x
    let n_bits_out = length y

    (x_head, x_tail) <- return (splitAt (n_bits_in - n_mask_bits) x)
    (y_head, y_tail) <- return (splitAt (n_bits_out - n_mask_bits) y)

    y_head <- mapUnary qnot y_head
    zipWithM_ (\xi yi -> qnot_at yi `controlled` xi) x_tail y_tail

    return (x_head ++ x_tail, y_head ++ y_tail)

-- Main function
main :: IO ()
main = do
    args <- getArgs
    let oracle = get_oracle (parse_args args)
    print_simple ASCII (period_circuit oracle)

parse_args :: [String] -> (String, Int, Int, Int)
parse_args [impl, n_bits_in_str, n_bits_out_str, n_mask_bits_str] =
    (impl, read n_bits_in_str, read n_bits_out_str, read n_mask_bits_str)
parse_args _ = error "usage: ./period synth|hand <n_bits_in> <n_bits_out> <n_mask_bits>"

get_oracle :: (String, Int, Int, Int) -> Oracle
get_oracle ("synth", n_bits_in, n_bits_out, n_mask_bits) =
    Oracle n_bits_in n_bits_out (synth_oracle (n_bits_out, n_mask_bits))
get_oracle ("hand", n_bits_in, n_bits_out, n_mask_bits) =
    Oracle n_bits_in n_bits_out (handwritten_oracle n_mask_bits)
get_oracle _ = error "oracle undefined"
