import sys
print("Python executable:", sys.executable)
try:
    import matplotlib
    print("matplotlib OK")
except Exception as e:
    print("matplotlib error:", e)

try:
    import pandas
    print("pandas OK")
except Exception as e:
    print("pandas error:", e)

try:
    import sklearn
    print("sklearn OK")
except Exception as e:
    print("sklearn error:", e)
