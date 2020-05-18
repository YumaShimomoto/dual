#
# @Author: kuroitu (2020)
# @email: Skuroitu@gmail.com
#
import numpy as np


from dual import Dual
import dual


def test():
    x = np.eye(2) * 2
    y = np.eye(3) * 3
    z = np.full(5, 5)
    print(dual.block([
        [x, np.zeros((2, 3))],
        [np.ones((3, 2)), y],
        [z]
        ]))


if __name__ == "__main__":
   test()
