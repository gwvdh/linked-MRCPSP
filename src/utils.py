from math import gcd


def get_earliest_start_time(n, T, M, R, E, p, L, r, VP):
    """
    Longest path calculation based on precedence relations. For each mode, the shortest processing time is taken. 
    E forms a DAG. 
    :param n: number of activities
    :param T: number of time slots 1,...,T
    :param M: number of modes
    :param R: List of resource capacities R[k]
    :param E: List of pairs of activity indices (i,j) indicating precedence relations
    :param p: List of processing times for each activity i in each mode m p[i][m]
    :param L: List of pairs of activity indices (i,j) indicating linked modes
    :param r: List of resource requirements for each activity i in each mode m on resource k r[i][m][k]
    :param VP: List of pairs of activity indices (i,j) that are not precedence-related
    :return: List of earliest start times for each activity i
    """
    earliest_starting_times = [0] * n
    current_activity = 0
    earliest_starting_times[current_activity] = 0
    for i in range(1,n):
        for j, k in E:
            if k == i:
                max_processing_time = min(p[j][m] for m in range(M))
                earliest_starting_times[i] = max(earliest_starting_times[j] + max_processing_time, earliest_starting_times[i])
    return earliest_starting_times



def get_latest_start_time(n, T, M, R, E, p, L, r, VP):
    """
    Longest path calculation based on precedence relations (reversed). For each mode, the shortest processing time is taken. 
    E forms a DAG. 
    :param n: number of activities
    :param T: number of time slots 1,...,T
    :param M: number of modes
    :param R: List of resource capacities R[k]
    :param E: List of pairs of activity indices (i,j) indicating precedence relations
    :param p: List of processing times for each activity i in each mode m p[i][m]
    :param L: List of pairs of activity indices (i,j) indicating linked modes
    :param r: List of resource requirements for each activity i in each mode m on resource k r[i][m][k]
    :param VP: List of pairs of activity indices (i,j) that are not precedence-related
    :return: List of latest start times for each activity i
    """
    earliest_starting_times = [-1] * n
    current_activity = n-1
    earliest_starting_times[current_activity] = 0
    for i in range(n-1, 0, -1):
        for k, j in E:
            if k == i:
                max_processing_time = max(p[j][m] for m in range(M))
                earliest_starting_times[i] = max(earliest_starting_times[j] + max_processing_time, earliest_starting_times[i])
    # reverse earliest_starting_times
    earliest_starting_times.reverse()
    return earliest_starting_times