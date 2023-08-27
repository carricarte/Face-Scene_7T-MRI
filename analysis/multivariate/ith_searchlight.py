import sys
import numpy as np
import pandas as pd

searchlight = sys.argv[1]
X = sys.argv[2]
y = sys.argv[3]
output_dir = sys.argv[4]

data = pd.read_csv(output_dir)
searchlight.fit(X, y)
data.loc[len(data)] = list(np.array(searchlight.scores_).reshape(-1))
data.to_csv(output_dir)
