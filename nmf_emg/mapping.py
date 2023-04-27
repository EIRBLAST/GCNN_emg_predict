import numpy as np


class AlphaMatrix:
    ar10 = np.array([[4.0504, 2.5198, 2.5198], [-0.0000, -1.5981, 1.1627], [0.7030, 1.5981, 1.1627]])

    ub = np.array([[3.4329, 2.0838, 2.0838], [0.0000, 1.5391, -1.5391], [-0.1432, 0.5975, 0.5975]])

    berrett = np.array([[3.4691, 3.5615, 2.1684], [0.3085, -0.0639, 2.1322], [-0.3248, 0.3085, 0.0717]])


class SynMatrix:
    ar10 = np.array(
        [
            [0.3386, 0.3536, -0.1017],
            [0.3386, 0.3536, -0.1017],
            [0.3386, 0.3536, -0.1017],
            [0.3386, 0.3536, -0.1017],
            [0.3386, -0.3536, -0.1017],
            [0.3386, -0.3536, -0.1017],
            [0.3386, -0.3536, -0.1017],
            [0.3386, -0.3536, -0.1017],
            [0.2649, -0.0000, 0.8819],
            [0.1122, -0.0000, 0.3735],
        ]
    )

    ub = np.array(
        [
            [0.3301, -0.0000, 0.6013],
            [0.0000, -0.0000, -0.0000],
            [0.2476, -0.0000, 0.4510],
            [0.2476, -0.0000, 0.4510],
            [0.0000, 0.0000, 0.0000],
            [0.2982, 0.3402, -0.1637],
            [0.2982, 0.3402, -0.1637],
            [0.1193, 0.1361, -0.0655],
            [0.0000, 0.0000, 0.0000],
            [0.2982, 0.3402, -0.1637],
            [0.2982, 0.3402, -0.1637],
            [0.1193, 0.1361, -0.0655],
            [0.0000, 0.0000, 0.0000],
            [0.2982, -0.3402, -0.1637],
            [0.2982, -0.3402, -0.1637],
            [0.1193, -0.1361, -0.0655],
            [0.0000, 0.0000, 0.0000],
            [0.2982, -0.3402, -0.1637],
            [0.2982, -0.3402, -0.1637],
            [0.1193, -0.1361, -0.0655],
        ]
    )

    berrett = np.array(
        [
            [-0.3973, -0.2624, -0.0963],
            [-0.3973, -0.2624, -0.0963],
            [-0.0000, 0.0000, 0.0000],
            [-0.0966, 0.6047, -0.4571],
            [0.39730, 0.2624, 0.0963],
            [0.39730, 0.2624, 0.0963],
            [0.07990, -0.1066, 0.7524],
            [0.42000, -0.4168, -0.3064],
            [0.42000, -0.4168, -0.3064],
        ]
    )