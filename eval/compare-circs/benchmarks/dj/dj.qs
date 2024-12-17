// Grabbed from microsoft andy
// filipw from github
// https://github.com/filipw/intro-to-qc-with-qsharp-book/blob/main/chapter-07/deutsch-jozsa/Program.qs
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Measurement;
open Microsoft.Quantum.Intrinsic;

function Format(result : Bool) : String {
    return result ? "constant" | "balanced";
}

operation DeutschJozsaAlgorithm(n : Int, oracle : ((Qubit[], Qubit) => Unit)) : Result[] {
    use (x, y) = (Qubit[n], Qubit());
    X(y);
    ApplyToEach(H, x);
    H(y);

    oracle(x, y);

    ApplyToEach(H, x);

    Reset(y);
    return MResetEachZ(x);
}

operation ConstantZero(x : Qubit[], y : Qubit) : Unit is Adj {
}

operation ConstantOne(x : Qubit[], y : Qubit) : Unit is Adj  {
    X(y);
}

operation BalancedEqual(x : Qubit[], y : Qubit) : Unit is Adj  {
    for qubit in x {
        CNOT(qubit, y);
    }
}

operation BalancedNotEqual(x : Qubit[], y : Qubit) : Unit is Adj {
    for qubit in x {
        CNOT(qubit, y);
    }
    X(y);
}

// @EntryPoint()
// operation Main() : Unit {
//     for n in [1, 6] {
//         let constZero = DeutschJozsaAlgorithm(n, ConstantZero);
//         let constOne = DeutschJozsaAlgorithm(n, ConstantOne);
//         let balEq = DeutschJozsaAlgorithm(n, BalancedEqual);
//         let balNotEq = DeutschJozsaAlgorithm(n, BalancedNotEqual);
//         Message($"n={n}");
//         Message($"Constant 0. Result: {Format(constZero)}.");
//         Message($"Constant 1. Result: {Format(constOne)}.");
//         Message($"Balanced. Result: {Format(balEq)}.");
//         Message($"Balanced opposite. Result: {Format(balNotEq)}.");
//     }
// }
