#!/usr/bin/env python3

import unittest

from integration_tests import IntegrationTests
from test_qwerty import RuntimeTests, ConvertAstTests
from test_qwerty_mlir import FileCheckTests

if __name__ == '__main__':
    unittest.main()
