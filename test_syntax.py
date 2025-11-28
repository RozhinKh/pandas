#!/usr/bin/env python
"""Simple syntax check for the sparse array module."""
import sys

try:
    import py_compile
    py_compile.compile('pandas/core/arrays/sparse/array.py', doraise=True)
    print("✓ Syntax check passed")
    sys.exit(0)
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)
