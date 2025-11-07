_________Qsharp's Qasm [Problem Size: 3]_________
OPENQASM 3.0;
include "stdgates.inc";
gate rxx(theta) q0, q1 { cx q0, q1; rx(theta) q0; cx q0, q1; }
gate ryy(theta) q0, q1 { cx q0, q1; ry(theta) q0; cx q0, q1; }
gate rzz(theta) q0, q1 { cx q0, q1; rz(theta) q0; cx q0, q1; }
qreg q[3];
creg c[3];
inv @ s q[0];
h q[0];
inv @ s q[1];
h q[1];
inv @ s q[2];
h q[2];
h q[0];
rz(0.7853981633974483) q[0];
ctrl(1) @ x q[1], q[0];
rz(-0.7853981633974483) q[0];
ctrl(1) @ x q[1], q[0];
rz(0.7853981633974483) q[1];
gphase(0.39269908169872414) ;
rz(0.39269908169872414) q[0];
ctrl(1) @ x q[2], q[0];
rz(-0.39269908169872414) q[0];
ctrl(1) @ x q[2], q[0];
rz(0.39269908169872414) q[2];
gphase(0.19634954084936207) ;
rz(0.4363323129985824) q[0];
measure q[0] -> c[0];
measure q[1] -> c[0];
measure q[2] -> c[0];

