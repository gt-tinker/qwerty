namespace QwertyCGO25.Eval.QsharpQirEmission {
    open Microsoft.Quantum.Arithmetic;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Synthesis;
    open Microsoft.Quantum.Intrinsic;

    operation PeriodFinding(n_in : Int, n_out: Int, oracle : ((Qubit[], Qubit[]) => Unit)) : Result[] {
        use (x, y) = (Qubit[n_in], Qubit[n_out]);
        ApplyToEach(H, x);

        oracle(x, y);

        // Account for QFT in Q# being little endian (bless their hearts for that
        // one). In a sense, this acts as the SWAPs in the typical Mike&Ike
        // implementation. Except we do it here since we are running QFT backwards
        //SwapReverseRegister(x);
        Adjoint ApplyQuantumFourierTransform(BigEndianAsLittleEndian(BigEndian(x)));

        ResetAll(y);
        return ForEach(MResetZ, x);
    }

    function PrepareOracle(numMaskBits : Int) : ((Qubit[], Qubit[]) => Unit) {
        return MaskingOracle(numMaskBits, _, _);
    }

    operation MaskingOracle(numMaskBits : Int, x : Qubit[], y : Qubit[]) : Unit is Adj + Ctl  {
        // Leading 11111111...1
        for i in 0 .. Length(y)-numMaskBits-1 {
            X(y[i]);
        }
        // Copy last numMaskBits bits from x to y
        for i in 0 .. numMaskBits-1 {
            CNOT(x[Length(x)-numMaskBits+i], y[Length(y)-numMaskBits+i]);
        }
    }

    @EntryPoint()
    operation Main() : Unit {
        let res = PeriodFinding(128, 127, PrepareOracle(64));
        Message($"Result: {res}.");
    }
}