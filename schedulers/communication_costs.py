
DELAY_BANDWIDTHS = {
     "Oregon-Virginia": (67, 0.79),
    "Oregon-Ohio": (49, 1.10),
    "Oregon-Tokyo": (96, 0.523),
    "Oregon-Seoul": (124, 0.46),
    "Oregon-Singapore": (163, 0.341),
    "Oregon-Sydney": (139, 0.36),
    "Oregon-London": (136, 0.42),
    "Oregon-Frankfurt": (143, 0.404),
    "Oregon-Ireland": (124, 0.482),
    "Virginia-Ohio": (11, 1.12),
    "Virginia-Tokyo": (143, 0.524),
    "Virginia-Seoul": (172, 0.500),
    "Virginia-Singapore": (230, 0.364),
    "Virginia-Sydney": (197, 0.383),
    "Virginia-London": (76, 1.16),
    "Virginia-Frankfurt": (90, 1.02),
    "Virginia-Ireland": (67, 1.05),
    "Ohio-Tokyo": (130, 0.694),
    "Ohio-Seoul": (159, 0.529),
    "Ohio-Singapore": (197, 0.452),
    "Ohio-Sydney": (185, 0.484),
    "Ohio-London": (86, 1.05),
    "Ohio-Frankfurt": (99, 0.799),
    "Ohio-Ireland": (77, 1.14),
    "Tokyo-Seoul": (34, 1.10),
    "Tokyo-Singapore": (73, 1.01),
    "Tokyo-Sydney": (100, 0.761),
    "Tokyo-London": (210, 0.366),
    "Tokyo-Frankfurt": (223, 0.36),
    "Tokyo-Ireland": (199, 0.465),
    "Seoul-Singapore": (74, 1.14),
    "Seoul-Sydney": (148, 0.58),
    "Seoul-London": (238, 0.342),
    "Seoul-Frankfurt": (235, 0.358),
    "Seoul-Ireland": (228, 0.335),
    "Singapore-Sydney": (92, 0.816),
    "Singapore-London": (169, 0.500),
    "Singapore-Frankfurt": (155, 0.535),
    "Singapore-Ireland": (179, 0.492),
    "Sydney-London": (262, 0.326),
    "Sydney-Frankfurt": (265, 0.328),
    "Sydney-Ireland": (254, 0.344),
    "London-Frankfurt": (14, 1.14),
    "London-Ireland": (12, 1.09),
    "Frankfurt-Ireland": (24, 1.08),
    "Sofia-Frankfurt":(25.05,0.320),
    "Sofia-Seoul":(323.03,0.356),
    "Sofia-Sydney":(298.90,0.268),
    "Sofia-Virginia":(120.46,0.286),
    "Sofia-Oregon":(114.24,0.282),
    "Sofia-Ohio":(125.96,0.284),
    "Sofia-London":(68.07,0.300),
    "Sofia-Tokyo":(318.88,0.260),
    "Sofia-Amsterdam":(38.24,0.300),
    "Amsterdam-Frankfurt":(8.72,0.300),
    "Amsterdam-Seoul":(288.39,0.268),
    "Amsterdam-Sydney":(265.94,0.270),
    "Amsterdam-Virginia":(80.81,0.300),
    "Amsterdam-Oregon":(72.29,0.300),
    "Amsterdam-Ohio":(75.31, 0.300),
    "Amsterdam-London":(7,0.300),
    "Amsterdam-Tokyo":(278.65,0.260),
    "Amsterdam-Singapore": (275,0.252),
    "Sofia-Singapore": (301, 0.241)
}
# measured per sample per layer:
COMPUTATIONAL_COST = {
    # encode heterogeneity through these values...
    # these were a bit higher due to us taking the maximum time a microbatch took
    # and as we had multiple nodes on the same gpu, there were additional delays
    "Sofia": 0.095,
    'Singapore':  0.095, 
    'Ireland':  0.095,  
    'Sydney':  0.095,
    'Frankfurt':  0.095, 
    'Seoul':  0.095,  
    'Oregon':  0.095, 
    'Ohio':  0.095,
    'Tokyo': 0.095,
    'Amsterdam': 0.095, 
    'Virginia':  0.095
}

def get_locations(setting = "geo-distributed"):
    if setting == "geo-distributed":
        return list(COMPUTATIONAL_COST.keys())
    elif setting == "single-cluster":
        return ["Seoul"]
    elif setting == "5-clusters":
        return ["Amsterdam", "Seoul", "Frankfurt", "Sydney", "Ohio"]
def id_to_loc(idx, setting="geo-distributed"):
    locs = get_locations(setting)
    idx = idx%len(locs)
    return locs[idx]
def get_computations(setting = "geo-distributed"):
    if setting == "geo-distributed":
        return COMPUTATIONAL_COST
    elif setting == "single-cluster":
        return {"Seoul": COMPUTATIONAL_COST["Seoul"]}
    elif setting == "5-clusters":
        ret = {}
        for loc in get_locations(setting):
            ret[loc] = COMPUTATIONAL_COST[loc]
        return ret

def delay_map(loc1,loc2, sz = 250*6291908):
    p1 = loc1
    p2 = loc2
    if DELAY_BANDWIDTHS.get(p1+"-"+p2) != None:
        ret = DELAY_BANDWIDTHS.get(p1+"-"+p2)
    elif DELAY_BANDWIDTHS.get(p2+"-"+p1) != None:
        ret = DELAY_BANDWIDTHS.get(p2+"-"+p1)
    else:
        ret = (10,0.80) # in the same cluster we can communicate in the order of GB/s, but it seems our cluster was achieving around 1 Gb/s on a good day
    
    # DT-FM communication costs were unrealistic (smallest one was in the order of 300MB/s), so we /2 to bring them to some realisting Mb/s...
    return ret[0]/1000 + 2*sz/(1024**3 * ret[1])

