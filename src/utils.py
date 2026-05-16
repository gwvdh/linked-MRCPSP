from math import gcd
from collections import defaultdict


def get_earliest_start_time(n, T, M, R, E, p, L, r, VP, ES=None):
    """
    Longest path calculation based on precedence relations. For each mode, the shortest processing time is taken. 
    E forms a DAG. 
    :param n: number of activities
    :param T: number of time slots 1,...,T
    :param M: number of modes for each activity
    :param R: List of resource capacities R[k]
    :param E: List of pairs of activity indices (i,j) indicating precedence relations
    :param p: List of processing times for each activity i in each mode m p[i][m]
    :param L: List of pairs of activity indices (i,j) indicating linked modes
    :param r: List of resource requirements for each activity i in each mode m on resource k r[i][m][k]
    :param VP: List of pairs of activity indices (i,j) that are not precedence-related
    :param ES: Earliest start time for each activity i
    :return: List of earliest start times for each activity i
    """
    preds: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, j in E:
        preds[j].append((i, min(p[i][m] for m in range(M[i]))))
    earliest_starting_times = [0] * n
    if ES is not None:
        for i in range(n):
            earliest_starting_times[i] = max(earliest_starting_times[i], ES[i])
    for i in range(n):
        for pred, min_p in preds[i]:
            earliest_starting_times[i] = max(earliest_starting_times[i], earliest_starting_times[pred] + min_p)
    return earliest_starting_times


def get_latest_start_time(n, T, M, R, E, p, L, r, VP):
    """
    Longest path calculation based on precedence relations (reversed). For each mode, the shortest processing time is taken. 
    E forms a DAG. 
    :param n: number of activities
    :param T: number of time slots 1,...,T
    :param M: number of modes for each activity
    :param R: List of resource capacities R[k]
    :param E: List of pairs of activity indices (i,j) indicating precedence relations
    :param p: List of processing times for each activity i in each mode m p[i][m]
    :param L: List of pairs of activity indices (i,j) indicating linked modes
    :param r: List of resource requirements for each activity i in each mode m on resource k r[i][m][k]
    :param VP: List of pairs of activity indices (i,j) that are not precedence-related
    :return: List of latest start times for each activity i
    """
    succs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, j in E:
        succs[i].append((j, min(p[j][m] for m in range(M[j]))))
    latest_starting_times = [T-1] * n
    for i in range(n-1, -1, -1):
        for succ, max_p in succs[i]:
            if latest_starting_times[succ] >= 0:
                latest_starting_times[i] = min(latest_starting_times[i], latest_starting_times[succ] - max_p)
    return latest_starting_times

