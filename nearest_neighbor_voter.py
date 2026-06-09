import math
from collections import defaultdict
import numpy as np
import pickle

def hungarian_algorithm(cost_matrix):
    """
    cost_matrix: 2D list or numpy 2D array (N x N)
    return: (row_ind, col_ind)
    """
    import math
   
    cost = [row[:] for row in cost_matrix]  # deep copy
    n = len(cost)

    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = -1

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # augmentation
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = []
    col_ind = []
    for j in range(1, n + 1):
        row_ind.append(p[j] - 1)
        col_ind.append(j - 1)
    return row_ind, col_ind


class NearestNeighborVoter:
    def __init__(self, candidate_positions, hungarian_mode=False):
        self.candidate_positions = candidate_positions
        self.hungarian_mode = hungarian_mode
        self.detections_list = []
        self.confidence_threshold = 0.8

    def update(self, detections):
        if not detections:
            return

        if self.hungarian_mode:
            self._update_hungarian(detections)
        else:
            for x, y, cls, conf in detections:
                self.detections_list.append((x, y, cls, conf))

    def _update_hungarian(self, detections):
        candidate_ids = list(self.candidate_positions.keys())
        candidate_coords = np.array([self.candidate_positions[cid] for cid in candidate_ids])

        det_coords = np.array([[x, y] for x, y, _, _ in detections])
        det_classes = [cls for _, _, cls, _ in detections]
        det_confs = [conf for _, _, _, conf in detections]

        cost_matrix = np.zeros((len(candidate_coords), len(det_coords)))
        for i, (cx, cy) in enumerate(candidate_coords):
            for j, (dx, dy) in enumerate(det_coords):
                cost_matrix[i, j] = math.hypot(dx - cx, dy - cy)

        row_ind, col_ind = hungarian_algorithm(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            x, y = det_coords[c]
            cls = det_classes[c]
            conf = det_confs[c]
            self.detections_list.append((x, y, cls, conf))

    def output(self):
        #print(self.candidate_positions)
        vote_map = {cid: defaultdict(float) for cid in self.candidate_positions}
        path = "/home/r1mini/catkin_ws/src/ROKAF_Autonomous_Car_2025/Maicon/object_coords_list.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.detections_list, f)
        for x, y, cls, conf in self.detections_list:
            if self.hungarian_mode:
                nearest_id = min(
                    self.candidate_positions.keys(),
                    key=lambda cid: math.hypot(x - self.candidate_positions[cid][0],
                                               y - self.candidate_positions[cid][1])
                )
            else:
                nearest_id = min(
                    self.candidate_positions.keys(),
                    key=lambda cid: math.hypot(x - self.candidate_positions[cid][0],
                                               y - self.candidate_positions[cid][1])
                )

            vote_map[nearest_id][cls] += conf

        result = {}
        for cid, class_scores in vote_map.items():
            if class_scores:
                best_cls = max(class_scores.items(), key=lambda x: x[1])[0]
                if class_scores[best_cls] >= self.confidence_threshold:
                    result[cid] = best_cls
                else:
                    result[cid] = None
            else:
                result[cid] = None

        return result

