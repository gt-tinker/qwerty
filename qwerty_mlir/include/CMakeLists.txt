# Ordering here is important: we want the QCirc libs later in the arguments
# to ld than the Qwerty libs. That's because the Qwerty libs will include
# qwutil, which depends on symbols in the QCirc libs
add_subdirectory(Qwerty)
add_subdirectory(QCirc)
