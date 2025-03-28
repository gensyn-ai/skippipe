
DELAY_BANDWIDTHS = {
    "Australia-Belgium":(129,0.00982666015625),
    "Australia-Canada":(98,0.01318359375),
    "Australia-Germany":(125.5,0.01046142578125),
    "Australia-Iowa":(84,0.01123046875),
    "Australia-Japan":(58,0.0137939453125),
    "Australia-LA":(69,0.0146484375),
    "Australia-NL":(128,0.009521484375),
    "Australia-Oregon":(68,0.0146484375),
    "Australia-Singapore":(48,0.02978515625),
    "Australia-Taiwan":(67,0.02099609375),
    "Australia-London":(132,0.0093994140625),
    "Belgium-Canada":(41,0.0198974609375),
    "Belgium-Germany":(4,0.2425),
    "Belgium-Iowa":(49,0.0283203125),
    "Belgium-Japan":(112,0.01171875),
    "Belgium-LA":(70,0.0194091796875),
    "Belgium-NL":(3,0.2425),
    "Belgium-Oregon":(66,0.0208740234375),
    "Belgium-Singapore":(83,0.01611328125),
    "Belgium-Taiwan":(105,0.012451171875),
    "Belgium-London":(3.5,0.2425),
    "Canada-Germany":(44,0.0318603515625),
    "Canada-Iowa":(14,0.07568359375),
    "Canada-Japan":(74,0.01806640625),
    "Canada-LA":(35,0.030517578125),
    "Canada-NL":(42,0.0250244140625),
    "Canada-Oregon":(30.5,0.047119140625),
    "Canada-Singapore": (107,0.01193847),
    "Canada-Taiwan":(89,0.01470947265625),
    "Canada-London":(38.5,0.036376953125),
    "Germany-Iowa":(53.5,0.026611328125),
    "Germany-Japan":(112,0.011083984375),
    "Germany-LA":(73,0.01824951171875),
    "Germany-NL":(3.5,0.2425),
    "Germany-Oregon":(68,0.0149169921875),
    "Germany-Singapore":(80,0.0125),
    "Germany-Taiwan":(102,0.0125732421875),
    "Germany-London":(6.6,0.18875),
    "Iowa-Japan":(72.5,0.01397705078125),
    "Iowa-LA":(23.4,0.06146240234375),
    "Iowa-NL":(51,0.02740478515625),
    "Iowa-Oregon":(19.3,0.09033203125),
    "Iowa-Taiwan": (88,0.0128),
    "Iowa-Singapore":(101.5,0.01263427734375),
    "Iowa-London":(48.1,0.0218505859375),
    "Japan-LA":(50,0.02777099609375),
    "Japan-NL":(108.5,0.011376953125),
    "Japan-Oregon":(46.2,0.03045654296875),
    "Japan-Singapore":(35.6,0.03900146484375),
    "Japan-Taiwan":(19.2,0.07672119140625),
    "Japan-London":(107,0.00888671875),
    "LA-NL":(71.5,0.014251708984375),
    "LA-Oregon":(12.4,0.1185302734375),
    "LA-Singapore":(79.5,0.016845703125),
    "LA-Taiwan":(64,0.016192626953125),
    "LA-London":(68,0.0149169921875),
    "NL-Oregon":(67,0.0203857421875),
    "NL-Singapore":(81.5,0.016357421875),
    "NL-Taiwan":(105.5,0.011962890625),
    "NL-London":(4.8,0.00023681640625),
    "Oregon-Singapore":(78.5,0.01678466796875),
    "Oregon-Taiwan":(59.5,0.0233154296875),
    "Oregon-London":(64,0.021240234375),
    "Singapore-Taiwan":(23.2,0.0618896484375),
    "Singapore-London":(85,0.01568603515625),
    "Taiwan-London":(109.5,0.0089599609375)

}
# measured per sample per layer:
COMPUTATIONAL_COST = {
    # encode heterogeneity through these values...
    # these were a bit higher due to us taking the maximum time a microbatch took
    # and as we had multiple nodes on the same gpu, there were additional delays
    "Australia": 0.055,
    'Belgium':  0.055, 
    'Canada':  0.055,  
    'Germany':  0.055,
    'Iowa':  0.055, 
    "Japan": 0.055,
    'LA':  0.055,  
    'NL':  0.055, 
    'Oregon':  0.055,
    'Singapore': 0.055,
    'Taiwan': 0.055, 
    'London':  0.055
}

def get_locations(setting = "geo-distributed"):
    if setting == "geo-distributed":
        return list(COMPUTATIONAL_COST.keys())
    elif setting == "single-cluster":
        return ["Germany"]
    elif setting == "5-clusters":
        return ["LA", "NL", "Germany", "Singapore", "Oregon"]
def id_to_loc(idx, setting="geo-distributed"):
    locs = get_locations(setting)
    idx = idx%len(locs)
    return locs[idx]
def get_computations(setting = "geo-distributed"):
    if setting == "geo-distributed":
        return COMPUTATIONAL_COST
    elif setting == "single-cluster":
        return {"Germany": COMPUTATIONAL_COST["Germany"]}
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
        # print("DONT HAVE",loc1,loc2)
        ret = (1,2.00) # in the same cluster we can communicate in the order of GB/s, but it seems our cluster was achieving around 1 Gb/s on a good day
    
    # print("Send",sz/(1024**2),"from",loc1,"to",loc2,ret[0]/1000 + sz/(ret[1]*1024**3))
    return ret[0]/1000 + sz/(ret[1]*1024**3)

