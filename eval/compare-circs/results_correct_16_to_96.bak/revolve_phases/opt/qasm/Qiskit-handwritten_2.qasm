_________Qiskit-handwritten's Qasm [Problem Size: 2]_________
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
u(0,0,-pi/2) q[0];
u(pi/2,0,pi) q[0];
u(pi/2,0,pi) q[0];
u(0,0,-pi/2) q[1];
u(pi/2,0,pi) q[1];
u(0,0,pi/4) q[1];
cx q[1],q[0];
u(0,0,-pi/4) q[0];
cx q[1],q[0];
u(0,0,pi/4) q[0];
u(0,0,0.4363323129985824) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
