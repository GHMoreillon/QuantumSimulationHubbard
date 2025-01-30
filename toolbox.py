# toolbox.py

def get_swap_sequence(initial, target):
    """
    Find the sequence of swap to map initial to target list
    
    Parameters:
    - initial (list) : the initial list.
    - target (list) : the target list. 

    Returns:
    - swaps (list) : a list containing pairs indicating swaps needed.
    """
    swaps = []
    current = initial[:]
    for i in range(len(current)):
        if current[i] != target[i]:
            target_index = current.index(target[i])
            swaps.append((i, target_index))
            current[i], current[target_index] = current[target_index], current[i]
    return swaps