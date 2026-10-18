import os
print("STEP 1: os imported", flush=True)
import sys
print("STEP 2: sys imported", flush=True)
import numpy as np
print("STEP 3: numpy imported", flush=True)
import pandas as pd
print("STEP 4: pandas imported", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print("STEP 5: matplotlib Agg imported", flush=True)

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
plt.savefig("test_headless.png")
plt.close()
print("STEP 6: successfully rendered test_headless.png", flush=True)
