// RUN: qwerty-opt %s | qwerty-opt | FileCheck --strict-whitespace %s
// Use strict whitespace because whitespace honestly makes a difference in
// readability, and this file's job is to test printers/parsers

// CHECK-LABEL: ccirc.circuit @flip(%arg0: !ccirc<wirebundle[3]>) rev {
//  CHECK-NEXT:   %0 = ccirc.not(%arg0) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
//  CHECK-NEXT:   ccirc.return %0 : !ccirc<wirebundle[3]>
//  CHECK-NEXT: }
ccirc.circuit @flip(%arg0: !ccirc<wirebundle[3]>) rev {
  %0 = ccirc.not(%arg0) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
  ccirc.return %0 : !ccirc<wirebundle[3]>
}

// CHECK-LABEL: ccirc.circuit private @const(%arg0: !ccirc<wirebundle[4]>) rev {
//  CHECK-NEXT:   %0 = ccirc.constant -3 : i4 : !ccirc<wirebundle[4]>
//  CHECK-NEXT:   %1 = ccirc.and(%arg0, %0) : (!ccirc<wirebundle[4]>, !ccirc<wirebundle[4]>) -> !ccirc<wirebundle[4]>
//  CHECK-NEXT:   ccirc.return %1 : !ccirc<wirebundle[4]>
//  CHECK-NEXT: }
ccirc.circuit private @const(%arg0: !ccirc<wirebundle[4]>) rev {
  // sign extended: 0b1101 | -1 << 4 == -3
  %0 = ccirc.constant -3 : i4 : !ccirc<wirebundle[4]>
  %1 = ccirc.and(%arg0, %0) : (!ccirc<wirebundle[4]>, !ccirc<wirebundle[4]>) -> !ccirc<wirebundle[4]>
  ccirc.return %1 : !ccirc<wirebundle[4]>
}
