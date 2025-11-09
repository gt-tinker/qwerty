_________Qwerty's Qasm [Problem Size: 2]_________
OPENQASM 3.0;
include "stdgates.inc";
qreg q[2];
creg c[2];
h q[0];
ctrl(1) @ x q[1], q[0];
tdg q[0];
ctrl(1) @ x q[1], q[0];
t q[0];
t q[1];
h q[1];
measure q[1] -> c[0];
measure q[0] -> c[1];

