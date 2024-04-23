# import os
# import hashlib

# import numpy as np


# def _hash_data(data):
#     return hashlib.sha256(np.ascontiguousarray(data)).hexdigest()


# def _save_to_disk(data, method, clustering):
#     hash = _hash_data(data)
#     file_name = f"clustering_cache/{method}_{hash}.npy"

#     np.save(file_name, clustering)


# def _load_from_disk(data, method):
#     hash = _hash_data(data)
#     file_name = f"clustering_cache/{method}_{hash}.npy"

#     if not os.path.exists(file_name):
#         return None

#     return np.load(file_name)
