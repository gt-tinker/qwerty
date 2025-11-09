_________Qsharp's Qasm [Problem Size: 2]_________
OPENQASM 3.0;
include "stdgates.inc";
gate rxx(theta) q0, q1 { cx q0, q1; rx(theta) q0; cx q0, q1; }
gate ryy(theta) q0, q1 { cx q0, q1; ry(theta) q0; cx q0, q1; }
gate rzz(theta) q0, q1 { cx q0, q1; rz(theta) q0; cx q0, q1; }
qreg q[2];
creg c[2];
h q[0];
rz(0.7853981633974483) q[0];
ctrl(1) @ x q[1], q[0];
rz(-0.7853981633974483) q[0];
ctrl(1) @ x q[1], q[0];
rz(0.7853981633974483) q[1];
gphase(0.39269908169872414) ;
measure q[0] -> c[0];
measure q[1] -> c[0];

