// RUN: qwerty-opt %s | qwerty-opt | FileCheck --strict-whitespace %s
// Use strict whitespace because whitespace honestly makes a difference in
// readability, and this file's job is to test printers/parsers

// CHECK-LABEL: ccirc.circuit @flip(%arg0: !ccirc<wirebundle[3]>) {
//  CHECK-NEXT:   %0 = ccirc.not(%arg0) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
//  CHECK-NEXT:   ccirc.return %0 : !ccirc<wirebundle[3]>
//  CHECK-NEXT: }
ccirc.circuit @flip(%arg0: !ccirc<wirebundle[3]>) {
  %0 = ccirc.not(%arg0) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
  ccirc.return %0 : !ccirc<wirebundle[3]>
}
