import numpy as np

def read(filename):

    n = np.fromfile(filename, count=1, dtype=np.int32)[0]
    start = np.fromfile(filename, count=n, dtype=np.uint32, offset=4)
    end = np.fromfile(filename, count=n, dtype=np.uint32, offset=(4 + 4 * n))
    smID = np.fromfile(filename, count=n, dtype=np.uint32, offset=(4 + 8 * n))
    blocks = np.fromfile(filename, count=n, dtype=np.uint32, offset=(4 + 12 * n))

    # Find the minimum time for each sm and subtract it from the start and end times
    for idx in smID:
        offset = np.min(start[smID == idx])
        start[smID == idx] -= offset
        end[smID == idx] -= offset
            
    return start, end, smID, blocks
