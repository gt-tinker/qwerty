add_subdirectory(CAPI)
add_subdirectory(Qwerty)
add_subdirectory(QCirc)
add_subdirectory(CCirc)

install(DIRECTORY Qwerty QCirc CCirc CAPI
    TYPE INCLUDE
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.td"
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Qwerty ${CMAKE_CURRENT_BINARY_DIR}/QCirc ${CMAKE_CURRENT_BINARY_DIR}/CCirc ${CMAKE_CURRENT_BINARY_DIR}/CAPI
    TYPE INCLUDE
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.td"
    PATTERN "*.inc"
)
