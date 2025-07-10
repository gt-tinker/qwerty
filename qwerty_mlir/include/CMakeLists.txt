add_subdirectory(CAPI)
# Ordering here is important: we want the QCirc libs later in the arguments
# to ld than the Qwerty libs. That's because the Qwerty libs will include
# qwutil, which depends on symbols in the QCirc libs
add_subdirectory(Qwerty)
add_subdirectory(QCirc)

install(DIRECTORY Qwerty QCirc CAPI
    TYPE INCLUDE
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.td"
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Qwerty ${CMAKE_CURRENT_BINARY_DIR}/QCirc ${CMAKE_CURRENT_BINARY_DIR}/CAPI
    TYPE INCLUDE
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.td"
    PATTERN "*.inc"
)
