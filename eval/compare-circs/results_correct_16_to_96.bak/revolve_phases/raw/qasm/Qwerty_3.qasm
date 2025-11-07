_________Qwerty's Qasm [Problem Size: 3]_________
OPENQASM 3.0;
include "stdgates.inc";
qreg q[3];
creg c[3];
sdg q[0];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
ctrl(1) @ x q[1], q[0];
tdg q[0];
ctrl(1) @ x q[1], q[0];
t q[0];
t q[1];
rz(0.392699) q[0];
ctrl(1) @ x q[2], q[0];
rz(-0.392699) q[0];
ctrl(1) @ x q[2], q[0];
rz(0.392699) q[2];
rz(0.436332) q[0];
measure q[1] -> c[0];
measure q[2] -> c[1];
measure q[0] -> c[2];

