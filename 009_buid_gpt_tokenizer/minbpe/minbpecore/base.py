def get_consecutive_pairs(ids):
    """ 
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    pair_count = {}
    for pair in zip(ids, ids[1:]):
        pair_count[pair] = pair_count.get(pair, 0) + 1
    return pair_count

def replace_pair(ids, replaceable_pair, replace_with_id):
    """ 
    In the list of integers (ids), replace all consecutive occurrences
    of replaceable_pair with the new integer token replace_with_id
    Example: ids=[1, 2, 3, 1, 2], replaceable_pair=(1, 2), replace_with_id=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and ids[i] == replaceable_pair[0] and ids[i+1] == replaceable_pair[1]:
            new_ids.append(replace_with_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
