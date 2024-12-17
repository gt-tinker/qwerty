// RUN: qwerty-opt -split-input-file -convert-qcirc-to-qir %s | qwerty-translate -split-input-file -mlir-to-qir | FileCheck %s

// CHECK-LABEL: define internal ptr @direct_call__trivial(ptr %0) {
//  CHECK-NEXT:   ret ptr %0
//  CHECK-NEXT: }
func.func private @direct_call__trivial(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
  return %arg0 : !qcirc<array<!qcirc.qubit>[2]>
}
// CHECK-EMPTY:

//  CHECK-NEXT: define ptr @direct_call(ptr %0) {
//  CHECK-NEXT:   %2 = call ptr @direct_call__trivial(ptr %0)
//  CHECK-NEXT:   ret ptr %2
//  CHECK-NEXT: }
func.func @direct_call(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
  %0 = call @direct_call__trivial(%arg0) : (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>
  return %0 : !qcirc<array<!qcirc.qubit>[2]>
}

// -----

// CHECK-LABEL: define internal ptr @direct_adj_call__trivial__adj(ptr %0) {
//  CHECK-NEXT:   ret ptr %0
//  CHECK-NEXT: }
func.func private @direct_adj_call__trivial__adj(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
  return %arg0 : !qcirc<array<!qcirc.qubit>[2]>
}
// CHECK-EMPTY:

//  CHECK-NEXT: define ptr @direct_adj_call(ptr %0) {
//  CHECK-NEXT:   %2 = call ptr @direct_adj_call__trivial__adj(ptr %0)
//  CHECK-NEXT:   ret ptr %2
//  CHECK-NEXT: }
func.func @direct_adj_call(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
  %0 = call @direct_adj_call__trivial__adj(%arg0) : (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>
  return %0 : !qcirc<array<!qcirc.qubit>[2]>
}

// -----

// CHECK-LABEL: @const_calli__trivial__metadata__func_table = internal constant [4 x ptr] [ptr @const_calli__trivial__metadata__fwd_stub, ptr null, ptr null, ptr null]
// CHECK-EMPTY:

//  CHECK-NEXT: define internal ptr @const_calli__trivial(ptr %0) {
//  CHECK-NEXT:   ret ptr %0
//  CHECK-NEXT: }
func.func private @const_calli__trivial(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
  return %arg0 : !qcirc<array<!qcirc.qubit>[2]>
}
// CHECK-EMPTY:

//  CHECK-NEXT: define internal void @const_calli__trivial__metadata__fwd_stub(ptr %0, ptr %1, ptr %2) {
//  CHECK-NEXT:   %4 = getelementptr inbounds { ptr }, ptr %1, i32 0, i32 0
//  CHECK-NEXT:   %5 = load ptr, ptr %4, align 8
//  CHECK-NEXT:   %6 = call ptr @const_calli__trivial(ptr %5)
//  CHECK-NEXT:   store ptr %6, ptr %2, align 8
//  CHECK-NEXT:   ret void
//  CHECK-NEXT: }
qcirc.callable_metadata private @const_calli__trivial__metadata captures [] specs [(fwd,0,@const_calli__trivial,(!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>)]
// CHECK-EMPTY:

//  CHECK-NEXT: define ptr @const_calli(ptr %0) {
//  CHECK-NEXT:   %2 = call ptr @__quantum__rt__callable_create(ptr @const_calli__trivial__metadata__func_table, ptr null, ptr null)
//  CHECK-NEXT:   %3 = call ptr @__quantum__rt__tuple_create(i64 ptrtoint (ptr getelementptr ({ ptr }, ptr null, i32 1) to i64))
//  CHECK-NEXT:   %4 = getelementptr inbounds { ptr }, ptr %3, i32 0, i32 0
//  CHECK-NEXT:   store ptr %0, ptr %4, align 8
//  CHECK-NEXT:   %5 = call ptr @__quantum__rt__tuple_create(i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64))
//  CHECK-NEXT:   call void @__quantum__rt__callable_invoke(ptr %2, ptr %3, ptr %5)
//  CHECK-NEXT:   call void @__quantum__rt__tuple_update_reference_count(ptr %3, i32 -1)
//  CHECK-NEXT:   %6 = load ptr, ptr %5, align 8
//  CHECK-NEXT:   call void @__quantum__rt__tuple_update_reference_count(ptr %5, i32 -1)
//  CHECK-NEXT:   call void @__quantum__rt__capture_update_reference_count(ptr %2, i32 -1)
//  CHECK-NEXT:   call void @__quantum__rt__callable_update_reference_count(ptr %2, i32 -1)
//  CHECK-NEXT:   ret ptr %6
//  CHECK-NEXT: }
func.func @const_calli(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
  %0 = qcirc.callable_create @const_calli__trivial__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
  %1 = qcirc.callable_invoke %0(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>, !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>
  return %1 : !qcirc<array<!qcirc.qubit>[2]>
}

// -----

// CHECK-LABEL: @const_adj_calli__trivial__metadata__func_table = internal constant [4 x ptr] [ptr null, ptr @const_adj_calli__trivial__metadata__adj_stub, ptr null, ptr null]
// CHECK-EMPTY:

//  CHECK-NEXT: define internal ptr @const_adj_calli__trivial__adj(ptr %0) {
//  CHECK-NEXT:   ret ptr %0
//  CHECK-NEXT: }
func.func private @const_adj_calli__trivial__adj(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
  return %arg0 : !qcirc<array<!qcirc.qubit>[2]>
}
// CHECK-EMPTY:

//  CHECK-NEXT: define internal void @const_adj_calli__trivial__metadata__adj_stub(ptr %0, ptr %1, ptr %2) {
//  CHECK-NEXT:   %4 = getelementptr inbounds { ptr }, ptr %1, i32 0, i32 0
//  CHECK-NEXT:   %5 = load ptr, ptr %4, align 8
//  CHECK-NEXT:   %6 = call ptr @const_adj_calli__trivial__adj(ptr %5)
//  CHECK-NEXT:   store ptr %6, ptr %2, align 8
//  CHECK-NEXT:   ret void
//  CHECK-NEXT: }
qcirc.callable_metadata private @const_adj_calli__trivial__metadata captures [] specs [(adj,0,@const_adj_calli__trivial__adj,(!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>)]
// CHECK-EMPTY:

//  CHECK-NEXT: define ptr @const_adj_calli(ptr %0) {
//  CHECK-NEXT:   %2 = call ptr @__quantum__rt__callable_create(ptr @const_adj_calli__trivial__metadata__func_table, ptr null, ptr null)
//  CHECK-NEXT:   call void @__quantum__rt__callable_make_adjoint(ptr %2)
//  CHECK-NEXT:   %3 = call ptr @__quantum__rt__tuple_create(i64 ptrtoint (ptr getelementptr ({ ptr }, ptr null, i32 1) to i64))
//  CHECK-NEXT:   %4 = getelementptr inbounds { ptr }, ptr %3, i32 0, i32 0
//  CHECK-NEXT:   store ptr %0, ptr %4, align 8
//  CHECK-NEXT:   %5 = call ptr @__quantum__rt__tuple_create(i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64))
//  CHECK-NEXT:   call void @__quantum__rt__callable_invoke(ptr %2, ptr %3, ptr %5)
//  CHECK-NEXT:   call void @__quantum__rt__tuple_update_reference_count(ptr %3, i32 -1)
//  CHECK-NEXT:   %6 = load ptr, ptr %5, align 8
//  CHECK-NEXT:   call void @__quantum__rt__tuple_update_reference_count(ptr %5, i32 -1)
//  CHECK-NEXT:   call void @__quantum__rt__capture_update_reference_count(ptr %2, i32 -1)
//  CHECK-NEXT:   call void @__quantum__rt__callable_update_reference_count(ptr %2, i32 -1)
//  CHECK-NEXT:   ret ptr %6
//  CHECK-NEXT: }
func.func @const_adj_calli(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
  %0 = qcirc.callable_create @const_adj_calli__trivial__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
  %1 = qcirc.callable_adj %0 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
  %2 = qcirc.callable_invoke %1(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>, !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>
  return %2 : !qcirc<array<!qcirc.qubit>[2]>
}
