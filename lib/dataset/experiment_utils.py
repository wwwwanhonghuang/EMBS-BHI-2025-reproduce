import numpy as np

def to_segment_sequence(microstate_sequence, keep_num_repetitions = False):
    pre_state = -1
    current_repetition = 0
    segment_sequence = []
    repetitions = []
    for i in range(len(microstate_sequence)):
        state = microstate_sequence[i]
        if pre_state < 0:
            pre_state = state
            current_repetition = 0
        elif microstate_sequence[i] != pre_state:
            segment_sequence.append(pre_state)
            repetitions.append(current_repetition)
            pre_state = state
            current_repetition = 0
        current_repetition += 1
    if keep_num_repetitions:
        return_value = (np.array(segment_sequence), np.array(repetitions))
        assert len(segment_sequence) == len(repetitions)
        return return_value
    return np.array(segment_sequence)
