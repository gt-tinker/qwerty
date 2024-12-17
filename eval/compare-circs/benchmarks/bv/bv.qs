// Grabbed from microsoft andy
// filipw from github
// https://github.com/filipw/intro-to-qc-with-qsharp-book/blob/qdk-1.0/chapter-07/bernstein-vazirani/Program.qs
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Diagnostics;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Measurement;
open Microsoft.Quantum.Intrinsic;

operation BernsteinVazirani(n : Int, oracle : ((Qubit[], Qubit) => Unit is Adj + Ctl)) : Result[] {
    use (x, y) = (Qubit[n], Qubit());
    X(y);
    ApplyToEach(H, x);
    H(y);

    oracle(x, y);

    ApplyToEach(H, x);

    Reset(y);
    return MResetEachZ(x);
}

function PrepareOracle(secret : Bool[]) : ((Qubit[],Qubit) => Unit is Adj + Ctl) {
    return Oracle(secret, _, _);
}

operation Oracle(secret : Bool[], x : Qubit[], y : Qubit) : Unit is Adj + Ctl  {
    for i in 0 .. Length(x) - 1 {
        if secret[i] {
            CNOT(x[i], y);
        }
    }
}
