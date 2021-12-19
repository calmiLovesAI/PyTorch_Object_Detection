

def get_aspect_ratio(cfg):
    ar = cfg["Model"]["aspect_ratio"]
    for i in range(len(ar)):
        for j in range(len(ar[i])):
            if isinstance(ar[i][j], str):
                ar[i][j] = eval(ar[i][j])
    return ar
