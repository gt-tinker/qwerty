_________Qsharp's Qasm [Problem Size: 4]_________
OPENQASM 3.0;
include "stdgates.inc";
gate rxx(theta) q0, q1 { cx q0, q1; rx(theta) q0; cx q0, q1; }
gate ryy(theta) q0, q1 { cx q0, q1; ry(theta) q0; cx q0, q1; }
gate rzz(theta) q0, q1 { cx q0, q1; rz(theta) q0; cx q0, q1; }
qreg q[4];
creg c[4];
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
rz(0.19634954084936207) q[0];
ctrl(1) @ x q[3], q[0];
rz(-0.19634954084936207) q[0];
ctrl(1) @ x q[3], q[0];
rz(0.19634954084936207) q[3];
gphase(0.09817477042468103) ;
measure q[0] -> c[0];
measure q[1] -> c[0];
measure q[2] -> c[0];
measure q[3] -> c[0];

