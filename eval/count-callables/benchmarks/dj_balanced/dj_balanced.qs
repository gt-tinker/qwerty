namespace QwertyCGO25.Eval.QsharpQirEmission {
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
        return ForEach(MResetZ, x);
    }

    operation BalancedEqual(x : Qubit[], y : Qubit) : Unit is Adj  {
        for qubit in x {
            CNOT(qubit, y);
        }
    }

    @EntryPoint()
    operation Main() : Unit {
        let balEq = DeutschJozsaAlgorithm(128, BalancedEqual);
        Message($"Balanced. Result: {balEq}.");
    }
}