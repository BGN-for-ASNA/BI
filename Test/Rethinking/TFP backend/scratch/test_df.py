import pandas as pd
import numpy as np

key = 's'
array = np.array([1, 2, 3])
all_params = []
all_params.append(pd.DataFrame({key: array}))
all_params.append(pd.DataFrame({f"{key}[0]": array}))

df_bi = pd.concat(all_params, axis=1)
print(df_bi.columns)
print(df_bi[['s[0]']])
