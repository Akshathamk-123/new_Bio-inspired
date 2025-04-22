'''import h5py
import hdf5plugin  # Must import this before accessing file!

path = r"F:\cRAIS\new_insect_inspired_navigation\train_dataset\zurich_city_11_a\events\left\events.h5"

with h5py.File(path, 'r') as f:
    x = f['events']['x']
    print("Compression:", x.compression)
    print("Trying to read a small slice...")
    print(x[:1000])  # This should now work if it was a compression issue
'''

import numpy as np
import h5py
import hdf5plugin

path = r"F:\cRAIS\new_insect_inspired_navigation\train_dataset\zurich_city_11_a\events\left\events.h5"

with h5py.File(path, 'r') as f:
    group = f['events']
    x = group['x'][:]
    y = group['y'][:]
    t = group['t'][:]
    p = group['p'][:]

    print(f"x shape: {x.shape}, dtype: {x.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    print(f"t shape: {t.shape}, dtype: {t.dtype}")
    print(f"p shape: {p.shape}, dtype: {p.dtype}")

    # Stack into structured format if needed
    events = np.stack([x, y, t, p], axis=1)  # Shape: (N, 4)
    print("Events stacked:", events.shape)
