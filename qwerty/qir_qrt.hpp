#ifndef QIR_QRT_H
#define QIR_QRT_H

#include <cstdint>

// Everything here is actually implemented in qir-runner. However, qir-runner
// is written in Rust, so we can't just #include a header file in that
// repository. They do have qir_stdlib.h, but it does not contain simulator
// calls

extern "C" {

struct QirArray;
struct QirQubit;
struct QirResult;
struct QirCallable;
typedef struct QirArray QirArray;
typedef struct QirQubit QirQubit;
typedef struct QirResult QirResult;
typedef struct QirCallable QirCallable;
typedef double QirFloat;
typedef void (*QirSpecFunc)(void *captures, void *args, void *result);
typedef void (*QirMemFunc)(void *captures, int32_t delta);

struct QirRotationArgs {
    double theta;
    QirQubit *qubit;
};
typedef struct QirRotationArgs QirRotationArgs;

void __quantum__rt__initialize(char *);
// 1Q gates
void __quantum__qis__x__body(QirQubit *);
void __quantum__qis__y__body(QirQubit *);
void __quantum__qis__z__body(QirQubit *);
void __quantum__qis__h__body(QirQubit *);
void __quantum__qis__rx__body(QirFloat, QirQubit *);
void __quantum__qis__ry__body(QirFloat, QirQubit *);
void __quantum__qis__rz__body(QirFloat, QirQubit *);
void __quantum__qis__s__body(QirQubit *);
void __quantum__qis__s__adj(QirQubit *);
void __quantum__qis__t__body(QirQubit *);
void __quantum__qis__t__adj(QirQubit *);
// 2Q gates
void __quantum__qis__cx__body(QirQubit *, QirQubit *);
void __quantum__qis__cy__body(QirQubit *, QirQubit *);
void __quantum__qis__cz__body(QirQubit *, QirQubit *);
// 3Q gates
void __quantum__qis__ccx__body(QirQubit *, QirQubit *, QirQubit *);
// NQ gates, with N > 2
void __quantum__qis__x__ctl(QirArray *, QirQubit *);
void __quantum__qis__y__ctl(QirArray *, QirQubit *);
void __quantum__qis__z__ctl(QirArray *, QirQubit *);
void __quantum__qis__h__ctl(QirArray *, QirQubit *);
void __quantum__qis__rx__ctl(QirArray *, QirRotationArgs *);
void __quantum__qis__ry__ctl(QirArray *, QirRotationArgs *);
void __quantum__qis__rz__ctl(QirArray *, QirRotationArgs *);
void __quantum__qis__s__ctl(QirArray *, QirQubit *);
void __quantum__qis__s__ctladj(QirArray *, QirQubit *);
void __quantum__qis__t__ctl(QirArray *, QirQubit *);
void __quantum__qis__t__ctladj(QirArray *, QirQubit *);
// Measurement/results
QirResult *__quantum__qis__m__body(QirQubit *);
void __quantum__qis__reset__body(QirQubit *);
QirResult *__quantum__rt__result_get_one();
bool __quantum__rt__result_equal();
// Allocation/deallocation
QirQubit *__quantum__rt__qubit_allocate();
void __quantum__rt__qubit_release(QirQubit *);
// Arrays
QirArray *__quantum__rt__array_create_1d(uint32_t, uint64_t);
QirArray *__quantum__rt__array_copy(QirArray *, bool);
void __quantum__rt__array_update_reference_count(QirArray *, int32_t);
void __quantum__rt__array_update_alias_count(QirArray *, int32_t);
void *__quantum__rt__array_get_element_ptr_1d(QirArray *, uint64_t);
uint64_t __quantum__rt__array_get_size_1d(QirArray *);
// Tuples (aka malloc(), apparently)
void *__quantum__rt__tuple_create(uint64_t);
void __quantum__rt__tuple_update_reference_count(void *, int32_t);
void __quantum__rt__tuple_update_alias_count(void *, int32_t);
// Callables
QirCallable *__quantum__rt__callable_create(QirSpecFunc (*)[4], QirMemFunc (*)[2], void *);
QirCallable *__quantum__rt__callable_copy(QirCallable *, bool);
void __quantum__rt__callable_invoke(QirCallable *, void *, void *);
void __quantum__rt__callable_make_adjoint(QirCallable *);
void __quantum__rt__callable_make_controlled(QirCallable *);
void __quantum__rt__callable_update_reference_count(QirCallable *, int32_t);
void __quantum__rt__callable_update_alias_count(QirCallable *, int32_t);
void __quantum__rt__capture_update_reference_count(QirCallable *, int32_t);
void __quantum__rt__capture_update_alias_count(QirCallable *, int32_t);
// Explosions
void __quantum__rt__fail(const char *);

} // extern "C"

#endif // QIR_QRT_H
