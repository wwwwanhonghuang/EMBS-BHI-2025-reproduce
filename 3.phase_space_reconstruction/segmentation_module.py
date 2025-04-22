from functools import cache 
import numpy as np
from tqdm import tqdm

class BaseSegmentGenerator(object):
    pass

class FiniteTimeDelayEEGSegmentGenerator(BaseSegmentGenerator):
    def __init__(self, data, time_delay = 15):
        self.data = data
        self.time_delay = time_delay
        
    def _time_delay_vector(self, t, time_delay):
        vec = self.data[max(t - time_delay + 1, 0): t + 1].ravel()
        return vec / np.linalg.norm(vec)

    def distance(self, t_i, t_j):
        time_delay = min(t_i, t_j, self.time_delay)
        time_delay_vector1 = self._time_delay_vector(t_i, time_delay)
        time_delay_vector2 = self._time_delay_vector(t_j, time_delay)           

        return np.sqrt(np.sum((time_delay_vector1 - time_delay_vector2) ** 2))
    
    def calculate_recurrent_plot_points(self, epsilon = 0.5):
        length = self.data.shape[0]
        
        recurrent_plot_points = []
        for i in tqdm(range(self.time_delay, length)):
            for j in range(i + 1, length):
                dist = self.distance(i, j)
                if dist < epsilon:
                    print(f'dist={dist} at coordination {i}, {j}')
                    if j - i >= 3:
                        recurrent_plot_points.append((i, j))
                    break
        return np.array(recurrent_plot_points)

class FiniteTimeDelaySegmentGenerator(BaseSegmentGenerator):
    def __init__(self, data, n_states, time_delay = 15, cut=[2, 800], data_with_repetition=False):
        self.time_delay = time_delay
        self.n_states = n_states
        self.cut = cut
        self.data_with_repetition = data_with_repetition
        if data_with_repetition:
            self.data = data[0]
            self.repetition = data[1]
        else:
            self.data = data
            self.repetition = None
            

        
    def _time_delay_vector(self, t, time_delay):
        vec = self.data[max(t - time_delay + 1, 0): t + 1]
        one_hot_vec = np.eye(self.n_states)[vec]
        return one_hot_vec / np.linalg.norm(one_hot_vec)

    def distance(self, t_i, t_j):
        time_delay = min(t_i, t_j, self.time_delay)
        time_delay_vector1 = self._time_delay_vector(t_i, time_delay)
        time_delay_vector2 = self._time_delay_vector(t_j, time_delay)           

        return np.sqrt(np.sum((time_delay_vector1 - time_delay_vector2) ** 2))
    
    def calculate_recurrent_segments(self, epsilon = 0.1):
        length = self.data.shape[0]
        
        recurrent_segments = []
        if self.data_with_repetition:
            repetition = []
        cut_lower = self.cut[0]
        cut_upper = self.cut[1] + 1
        for i in tqdm(range(self.time_delay, length)):
            for j in range(i + 1, length):
                dist = self.distance(i, j)
                if dist < epsilon:
                    if j - i > self.cut[0] and j - i <= cut_upper:
                        recurrent_segments.append(self.data[max(i - self.time_delay + 1, 0): j + 1])
                        repetition.append(self.repetition[max(i - self.time_delay + 1, 0): j + 1])
                    break
        if self.data_with_repetition:
            return (recurrent_segments, repetition)
        return recurrent_segments
    
    def calculate_recurrent_plot_points(self, epsilon = 0.1):
        length = self.data.shape[0]
        
        recurrent_plot_points = []
        cut_lower = self.cut[0]
        cut_upper = self.cut[1] + 1
        for i in tqdm(range(self.time_delay, length)):
            for j in range(i + 1, length):
                dist = self.distance(i, j)
                if dist < epsilon:
                    if j - i > self.cut[0] and j - i <= cut_upper:
                        print(f'dist={dist} at coordination {i}, {j}')
                        recurrent_plot_points.append((i, j))
                    break
        return np.array(recurrent_plot_points)

class InfinitePhaseSpaceReonstructionBasedSegmentGenerator(BaseSegmentGenerator):
    def __init__(self, data, arg_lambda = 0.5, truncation = -1):
        self.arg_lambda = arg_lambda
        self.truncation = truncation
        self.data = data
        
    def _observation_distance_l1(self, t_i, t_j):
        return np.sum(np.abs(self.data[t_i] - self.data[t_j]))
    
    def _distance(self, t_i, t_j, dept):
        _observation_differs = self._observation_distance_l1(t_i, t_j)
        recursive_distance = 0 \
            if t_i == 0 or t_j == 0 or (self.truncation > 0 and dept >= self.truncation) \
            else self.arg_lambda * 1 # self._distance(t_i - 1, t_j - 1, dept + 1)
        distance = _observation_differs + recursive_distance
        return distance

    @cache
    def distance(self, t_i, t_j):
        return self._distance(t_i, t_j, 0)
    
    def calculate_recurrent_plot_points(self, epsilon = 0.01):
        length = self.data.shape[0]
        
        recurrent_plot_points = []
        for i in tqdm(range(length)):
            for j in range(length):
                dist = self.distance(i, j)
                print(dist)
                if dist < epsilon:
                    recurrent_plot_points.append((i, j))
                    break
        return np.array(recurrent_plot_points)
