namespace QwertyCGO25.Eval.QsharpQirEmission {
    // Grabbed from microsoft andy
    // filipw from github
    // https://github.com/filipw/intro-to-qc-with-qsharp-book/blob/qdk-1.0/chapter-07/grover/Program.qs
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math;

    operation Grover(n : Int, r: Int, oracle : Qubit[] => Unit) : Result[] {
        //let r = Floor(PI() / 4.0 * Sqrt(IntAsDouble(2 ^ n)));
        use qubits = Qubit[n];

        ApplyToEach(H, qubits);

        // Grover iterations
        for x in 1 .. r {

            // oracle
            oracle(qubits);

            // diffusor
            Diffusion(qubits);
        }

        let result = MeasureEachZ(qubits);
        ResetAll(qubits);
        return result;
    }

    operation Oracle(qubits : Qubit[]) : Unit {
        Controlled Z(Most(qubits), Tail(qubits));
    }

    operation Diffusion(qubits : Qubit[]) : Unit {
        within {
            ApplyToEachA(H, qubits);
            ApplyToEachA(X, qubits);
        } apply {
            Controlled Z(Most(qubits), Tail(qubits));
        }
    }

    @EntryPoint()
    operation Main() : Unit {
        let res = Grover(128, 12, Oracle);
        Message($"Result: {res}.");
    }
}