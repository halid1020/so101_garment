# Makes test/ a regular package so `python -m unittest test.test_*` resolves
# here instead of the Python stdlib's `test` package (regular packages win
# over namespace packages during import resolution).
