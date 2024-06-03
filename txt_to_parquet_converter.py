"""Converts output.txt files to new parquet format. Saves parquet file in the folder of provided output.txt.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

with open(sys.argv[1]) as f:
    lines = np.array(f.read().split("\n"))[1:-1]

sequences = list(lines[::2])
svs = list(lines[1::2])

pred_intensities = [float(line.split()[-1]) for line in sequences]
charge = [int(line.split()[-2]) for line in sequences]
energy = [float(line.split()[-3]) for line in sequences]
seq_list = [line.split()[:-3] for line in sequences]

shap_values_list = [[float(m) for m in line.split()] for line in svs]
output = {
    "sequence": seq_list,
    "shap_values": shap_values_list,
    "intensity": pred_intensities,
    "energy": energy,
    "charge": charge,
}
pd.DataFrame(output).to_parquet(
    str(Path(sys.argv[1]).parent.absolute()) + "/output.parquet.gzip",
    compression="gzip",
)
