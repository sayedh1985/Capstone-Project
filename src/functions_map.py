# src/function_maps.py
FUNCTION_DIMENSIONS = {
    1: 2,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 5,
    7: 6,
    8: 8
}

def get_dim(fid):
    return FUNCTION_DIMENSIONS.get(fid, None)
