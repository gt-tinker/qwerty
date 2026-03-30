// RUN: qwerty-opt -convert-ccirc-to-xag -split-input-file %s | FileCheck %s

// CHECK-LABEL: ccirc.circuit @xor_to_parity(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
//  CHECK-NEXT:   %0 = ccirc.parity(%arg0, %arg1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %0 : !ccirc<wire[1]>
//  CHECK-NEXT: }
ccirc.circuit @xor_to_parity(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
  %0 = ccirc.xor(%arg0, %arg1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %0 : !ccirc<wire[1]>
}

// -----

// CHECK-LABEL: ccirc.circuit @or_to_demorgan(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
//  CHECK-NEXT:   %0 = ccirc.not(%arg0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %1 = ccirc.not(%arg1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %2 = ccirc.and(%0, %1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.not(%2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %3 : !ccirc<wire[1]>
//  CHECK-NEXT: }
ccirc.circuit @or_to_demorgan(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>) irrev {
  %0 = ccirc.or(%arg0, %arg1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %0 : !ccirc<wire[1]>
}

// -----

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

// CHECK-LABEL: ccirc.circuit @multi_bit_parity(%arg0: !ccirc<wire[2]>, %arg1: !ccirc<wire[2]>) irrev {
//  CHECK-NEXT:   %0:2 = ccirc.wireunpack %arg0 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1:2 = ccirc.wireunpack %arg1 : (!ccirc<wire[2]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %2 = ccirc.parity(%0#0, %1#0) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.parity(%0#1, %1#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %4 = ccirc.wirepack(%2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[2]>
//  CHECK-NEXT:   ccirc.return %4 : !ccirc<wire[2]>
//  CHECK-NEXT: }
ccirc.circuit @multi_bit_parity(%arg0: !ccirc<wire[2]>, %arg1: !ccirc<wire[2]>) irrev {
  %0 = ccirc.parity(%arg0, %arg1) : (!ccirc<wire[2]>, !ccirc<wire[2]>) -> !ccirc<wire[2]>
  ccirc.return %0 : !ccirc<wire[2]>
}

// -----

// Multi-bit XOR: split into element-wise XORs (Round One), then each
// single-bit XOR becomes a parity (Round Two).
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

// Multi-bit OR: split into element-wise ORs (Round One), then each
// single-bit OR becomes not(and(not, not)) via De Morgan's law (Round Two).
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
