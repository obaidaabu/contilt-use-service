import re
def pos_name_mapper(pos_name):
    pos_name = pos_name.lower()
    if pos_name == "adj":
        return "adjective"
    if pos_name == "noun":
        return "noun"
    if pos_name == "propn":
        return "propernoun"
    if pos_name == "cconj":
        return "conjunction"
    if pos_name == "verb":
        return "verb"
    if pos_name == "det":
        return "determiner"
    if pos_name == "adv":
        return "adverb"
    if pos_name == "sconj":
        return "conjunction"


def softmax(x):
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)


def average(lst):
    return sum(lst) / len(lst)


def tupleToScoreMap(tupleList):
    res = {}
    for itm in tupleList:
        res[itm[0]] = itm[1]
    return res


def fix_sentence(sent):
    sent = re.sub(r"[\s\,]+\.[\s\,]*", ". ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    sent = re.sub(r"\s+,", ",",sent)
    sent = re.sub(r" an\s+(?=[^auoie])", " a ",sent)
    sent = re.sub(r" a\s+(?=[auoie])", " an ",sent)
    sent = re.sub(r"and\s+and", "and", sent)
    sent = re.sub(r"(\s*,\s*){2,}", ", ", sent)
    return sent

def scoreMapToPercentile(scoreMap):
    valuesSet = set([scoreMap[a] for a in  scoreMap])
    sortedValues = sorted([v for v in valuesSet])
    valuesToPercentage = {}
    i = 0
    lenTotal = len(sortedValues)
    for v in sortedValues:
        valuesToPercentage[v] = (i+1)/lenTotal
        i = i+1

    finalres = {}
    for k in scoreMap:
        finalres[k] = valuesToPercentage[scoreMap[k]]
    return finalres

def scoreMapToMinMaxNormalized(scoreMap):
    min_val = min([scoreMap[x] for x in scoreMap])
    max_val = max([scoreMap[x] for x in scoreMap])
    res = {}
    for k in scoreMap:
        res[k] = (scoreMap[k]-min_val+0.001)/(max_val-min_val+0.001)
    return res

def scoreSmartNormalize(scoreMap):
    norm1 = scoreMapToPercentile(scoreMap)
    norm2 = scoreMapToMinMaxNormalized(scoreMap)
    res = {}
    for k in norm1:
        res[k] = (norm1[k]**0.7)*(norm2[k]**0.3)
    return res

def scoreMapToPercentileOld(scoreMap):
    keylist = [a for a in  scoreMap]
    sortedKeyList = sorted(keylist, key=lambda x: scoreMap[x], reverse=True)
    res = {}
    i = 0
    lenTotal = len(sortedKeyList)
    for k in sortedKeyList:
        res[k] = (lenTotal-i)/lenTotal
        i = i+1
    return res


def getValueOrDefault(d, k, default_v):
    if k not in d:
        return default_v
    return d[k]

def addVToMap(map_param, k, v):
    if k not in map_param:
        map_param[k] = v
    else:
        map_param[k] = map_param[k]+v

def normalizeSimMatrix(simMatrix):
    keyMaxSim = {}
    keyMinSim = {}
    res = {}
    for k in simMatrix:
        keyMaxSim[k] = max([v for v in simMatrix[k].values() if v < 0.99])

        keyMinSim[k] = min(sorted(simMatrix[k].values(), key=lambda k: k, reverse=True)[0:max(int(len(simMatrix[k]) * 0.2), len(simMatrix[k]))])

    for k1 in simMatrix:
        res[k1] = {}
        for k2 in simMatrix[k1]:
            denominator = (max(keyMaxSim[k1], keyMaxSim[k2])-min(keyMinSim[k1], keyMinSim[k2]))
            if denominator == 0:
                res[k1][k2] = 0
            else:
                res[k1][k2] = max(min(1.0, (simMatrix[k1][k2]-min(keyMinSim[k1], keyMinSim[k2]))/denominator), 0)
    return res
