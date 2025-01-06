import numpy as np

def to_segment_sequence(microstate_sequence):
    pre_state = -1
    segment_sequence = []
    for i in range(len(microstate_sequence)):
        state = microstate_sequence[i]
        if pre_state < 0:
            pre_state = state
        elif microstate_sequence[i] != pre_state:
            segment_sequence.append(pre_state)
            pre_state = state
    return np.array(segment_sequence)