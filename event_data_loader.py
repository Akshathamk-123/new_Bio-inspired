import h5py
import numpy as np
import hdf5plugin

'''def load_event_data(events_path):
    print(f"Loading event data from: {events_path}")
    try:
        with h5py.File(events_path, 'r') as f:
            group = f['events']
            print("Checking contents of 'events' group:")
            for key in group:
                item = group[key]
                print(f"  - {key}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")

            try:
                # Check if all required keys are datasets
                if not all(isinstance(group[key], h5py.Dataset) for key in ['x', 'y', 'p', 't']):
                    raise RuntimeError("One or more of 'x', 'y', 'p', 't' is not a Dataset.")

                # Attempt to read datasets
                x = group['x'][:]
                y = group['y'][:]
                p = group['p'][:]
                t = group['t'][:]

                print(f"Shapes: x={x.shape}, y={y.shape}, p={p.shape}, t={t.shape}")
                data = np.stack([x, y, p, t], axis=1)
                return data
            except Exception as e:
                print(f"Partial access failed: {e}")
                raise RuntimeError("Could not read event data from datasets.")

    except OSError as e:
        print(f"Error opening file {events_path}: {e}")
        raise RuntimeError("Failed to open event HDF5 file.")
'''

def load_event_data(events_path):
    print(f"Loading event data from: {events_path}")
    try:
        with h5py.File(events_path, 'r') as f:
            # Check the available groups
            print("Available groups in file:", list(f.keys()))
            
            if 'events' not in f:
                raise KeyError("'events' group not found in file.")

            group = f['events']
            print("Checking contents of 'events' group:")
            for key in group:
                item = group[key]
                print(f"  - {key}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")

            # Proceed if all required keys are datasets
            required_keys = ['x', 'y', 'p', 't']
            for key in required_keys:
                if key not in group or not isinstance(group[key], h5py.Dataset):
                    raise KeyError(f"Dataset '{key}' not found or is not a Dataset.")
            
            # Read datasets
            x = group['x'][:1000]
            y = group['y'][:1000]
            p = group['p'][:1000]
            t = group['t'][:1000]

            print(f"Shapes: x={x.shape}, y={y.shape}, p={p.shape}, t={t.shape}")
            data = np.stack([x, y, p, t], axis=1)
            return data

    except KeyError as e:
        print(f"Error: {e}")
        raise RuntimeError("Error accessing required datasets.")
    except OSError as e:
        print(f"Error opening file {events_path}: {e}")
        raise RuntimeError("Failed to open event HDF5 file.")


def load_rectify_map(map_path):
    print(f"Loading rectify map from: {map_path}")
    try:
        with h5py.File(map_path, 'r') as f:
            rectify_map = f['rectify_map'][:]
        return rectify_map
    except Exception as e:
        print(f"Error loading rectify map: {e}")
        raise RuntimeError("Could not load rectify map.")

def preprocess_events(events, rectify_map=None):
    print("Preprocessing events...")
    if rectify_map is not None:
        # Apply spatial correction if needed
        print("Rectify map found but no processing logic yet.")
        pass
    return events
