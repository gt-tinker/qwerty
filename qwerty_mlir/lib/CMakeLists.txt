add_subdirectory(CAPI)
# Ordering here is important: we want the Quantum libs later in the arguments
# to ld than the Qwerty libs. That's because the Qwerty libs will include
# qwutil, which depends on symbols in the Quantum libs
add_subdirectory(Qwerty)
add_subdirectory(QCirc)
add_subdirectory(CCirc)
