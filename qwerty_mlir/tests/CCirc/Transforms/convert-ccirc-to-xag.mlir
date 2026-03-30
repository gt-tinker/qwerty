// RUN: qwerty-opt -convert-ccirc-to-xag -split-input-file %s | FileCheck %s

// Single-bit XOR test
// CHECK-LABEL: ccirc.circuit @single_bit_xor(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
//  CHECK-NEXT:   %0 = ccirc.parity(%arg0, %arg1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %0 : !ccirc<wire[1]>
//  CHECK-NEXT: }
ccirc.circuit @single_bit_xor(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
  %0 = ccirc.xor(%arg0, %arg1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %0 : !ccirc<wire[1]>
}

// -----

// Single-bit OR test
// CHECK-LABEL: ccirc.circuit @single_bit_or(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
//  CHECK-NEXT:   %0 = ccirc.not(%arg0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %1 = ccirc.not(%arg1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %2 = ccirc.and(%0, %1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.not(%2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %3 : !ccirc<wire[1]>
//  CHECK-NEXT: }
ccirc.circuit @single_bit_or(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
  %0 = ccirc.or(%arg0, %arg1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %0 : !ccirc<wire[1]>
}

// -----

// Single-bit AND test
// CHECK-LABEL: ccirc.circuit @single_bit_and(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
//  CHECK-NEXT:   %0 = ccirc.and(%arg0, %arg1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %0 : !ccirc<wire[1]>
//  CHECK-NEXT: }
ccirc.circuit @single_bit_and(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
  %0 = ccirc.and(%arg0, %arg1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %0 : !ccirc<wire[1]>
}

// -----

// Multi-bit NOT test
// CHECK-LABEL: ccirc.circuit @multi_bit_not(%arg0: !ccirc<wire[2]>) irrev {
//  CHECK-NEXT:   %0:2 = ccirc.wireunpack %arg0 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1 = ccirc.not(%0#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %2 = ccirc.not(%0#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.wirepack(%1, %2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[2]>
//  CHECK-NEXT:   ccirc.return %3 : !ccirc<wire[2]>
//  CHECK-NEXT: }
ccirc.circuit @multi_bit_not(%arg0: !ccirc<wire[2]>) irrev {
  %0 = ccirc.not(%arg0) : (!ccirc<wire[2]>) -> !ccirc<wire[2]>
  ccirc.return %0 : !ccirc<wire[2]>
}

// -----

// Multi-bit AND test
// CHECK-LABEL: ccirc.circuit @multi_bit_and(%arg0: !ccirc<wire[2]>, %arg1: !ccirc<wire[2]>) irrev {
//  CHECK-NEXT:   %0:2 = ccirc.wireunpack %arg0 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1:2 = ccirc.wireunpack %arg1 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %2 = ccirc.and(%0#0, %1#0) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.and(%0#1, %1#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %4 = ccirc.wirepack(%2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[2]>
//  CHECK-NEXT:   ccirc.return %4 : !ccirc<wire[2]>
//  CHECK-NEXT: }
ccirc.circuit @multi_bit_and(%arg0: !ccirc<wire[2]>, %arg1: !ccirc<wire[2]>) irrev {
  %0 = ccirc.and(%arg0, %arg1) : (!ccirc<wire[2]>, !ccirc<wire[2]>) -> !ccirc<wire[2]>
  ccirc.return %0 : !ccirc<wire[2]>
}

// -----

// Multi-bit XOR test
// CHECK-LABEL: ccirc.circuit @multi_bit_xor(%arg0: !ccirc<wire[2]>, %arg1: !ccirc<wire[2]>) irrev {
//  CHECK-NEXT:   %0:2 = ccirc.wireunpack %arg0 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1:2 = ccirc.wireunpack %arg1 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %2 = ccirc.parity(%0#0, %1#0) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.parity(%0#1, %1#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %4 = ccirc.wirepack(%2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[2]>
//  CHECK-NEXT:   ccirc.return %4 : !ccirc<wire[2]>
//  CHECK-NEXT: }
ccirc.circuit @multi_bit_xor(%arg0: !ccirc<wire[2]>, %arg1: !ccirc<wire[2]>) irrev {
  %0 = ccirc.xor(%arg0, %arg1) : (!ccirc<wire[2]>, !ccirc<wire[2]>) -> !ccirc<wire[2]>
  ccirc.return %0 : !ccirc<wire[2]>
}

// -----

// Multi-bit OR test
// CHECK-LABEL: ccirc.circuit @multi_bit_or(%arg0: !ccirc<wire[2]>, %arg1: !ccirc<wire[2]>) irrev {
//  CHECK-NEXT:   %0:2 = ccirc.wireunpack %arg0 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1:2 = ccirc.wireunpack %arg1 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %2 = ccirc.not(%0#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.not(%1#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %4 = ccirc.and(%2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %5 = ccirc.not(%4) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %6 = ccirc.not(%0#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %7 = ccirc.not(%1#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %8 = ccirc.and(%6, %7) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %9 = ccirc.not(%8) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %10 = ccirc.wirepack(%5, %9) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[2]>
//  CHECK-NEXT:   ccirc.return %10 : !ccirc<wire[2]>
//  CHECK-NEXT: }
ccirc.circuit @multi_bit_or(%arg0: !ccirc<wire[2]>, %arg1: !ccirc<wire[2]>) irrev {
  %0 = ccirc.or(%arg0, %arg1) : (!ccirc<wire[2]>, !ccirc<wire[2]>) -> !ccirc<wire[2]>
  ccirc.return %0 : !ccirc<wire[2]>
}

// -----

// Example from PR
// CHECK-LABEL: ccirc.circuit private @foo_0(%arg0: !ccirc<wire[3]>, %arg1: !ccirc<wire[3]>, %arg2: !ccirc<wire[3]>) irrev {
//  CHECK-NEXT:   %0:3 = ccirc.wireunpack %arg0 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1:3 = ccirc.wireunpack %arg1 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %2 = ccirc.parity(%0#0, %1#0) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.parity(%0#1, %1#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %4 = ccirc.parity(%0#2, %1#2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %5 = ccirc.wirepack(%2, %3, %4) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
//  CHECK-NEXT:   %6:3 = ccirc.wireunpack %5 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %7:3 = ccirc.wireunpack %arg2 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %8 = ccirc.parity(%6#0, %7#0) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %9 = ccirc.parity(%6#1, %7#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %10 = ccirc.parity(%6#2, %7#2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %11 = ccirc.wirepack(%8, %9, %10) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
//  CHECK-NEXT:   %12 = ccirc.constant false : !ccirc<wire[1]>
//  CHECK-NEXT:   %13 = ccirc.constant false : !ccirc<wire[1]>
//  CHECK-NEXT:   %14 = ccirc.constant true : !ccirc<wire[1]>
//  CHECK-NEXT:   %15 = ccirc.wirepack(%12, %13, %14) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
//  CHECK-NEXT:   %16:3 = ccirc.wireunpack %11 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %17:3 = ccirc.wireunpack %15 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %18 = ccirc.not(%16#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %19 = ccirc.not(%17#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %20 = ccirc.and(%18, %19) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %21 = ccirc.not(%20) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %22 = ccirc.not(%16#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %23 = ccirc.not(%17#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %24 = ccirc.and(%22, %23) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %25 = ccirc.not(%24) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %26 = ccirc.not(%16#2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %27 = ccirc.not(%17#2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %28 = ccirc.and(%26, %27) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %29 = ccirc.not(%28) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %30 = ccirc.wirepack(%21, %25, %29) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
//  CHECK-NEXT:   ccirc.return %30 : !ccirc<wire[3]>
//  CHECK-NEXT: }
ccirc.circuit private @foo_0(%arg0: !ccirc<wire[3]>, %arg1: !ccirc<wire[3]>, %arg2: !ccirc<wire[3]>) irrev {
  %0 = ccirc.xor(%arg0, %arg1) : (!ccirc<wire[3]>, !ccirc<wire[3]>) -> !ccirc<wire[3]>
  %1 = ccirc.xor(%0, %arg2) : (!ccirc<wire[3]>, !ccirc<wire[3]>) -> !ccirc<wire[3]>
  %2 = ccirc.constant 1 : i3 : !ccirc<wire[3]>
  %3 = ccirc.or(%1, %2) : (!ccirc<wire[3]>, !ccirc<wire[3]>) -> !ccirc<wire[3]>
  ccirc.return %3 : !ccirc<wire[3]>
}