calc_all_libs()

add_executable(qwerty-test support/qwerty-test.cpp
                           support/test_support.cpp
                           test_ast.cpp
                           ast_visitor/test_canonicalize.cpp
                           ast_visitor/test_type_checking.cpp
                           ast_visitor/test_flag_immat.cpp)
target_include_directories(qwerty-test PRIVATE ${GTEST_INCLUDE_PATH} support/)
target_link_libraries(qwerty-test ${GTEST_LIB_PATH} qwc ${all_libs})
install(TARGETS qwerty-test DESTINATION ${SKBUILD_SCRIPTS_DIR})

add_test(qwerty-test qwerty-test)
