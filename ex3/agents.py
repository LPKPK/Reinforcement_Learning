import numpy as np
import env
import matplotlib.pyplot as plt

# class gridAgent():

#     def __init__(self) -> None:

#         self.delta = 0
#         self.V_s = np.zeros((5, 5), dtype=float)    
    
#     def Q5(self, theta = 0.001, gamma = 0.9):
#         #   
#         while self.delta > theta:

if __name__ == '__main__':

    # # Q5 (a)
    # gridW = env.Gridworld5x5()
    # print("Q5 (a)\n", gridW.Q5_a())
    # # Q5 (b)
    # v,pi = gridW.Q5_b()
    # print("Q5 (b)\n", v, '\n',pi)
    # # Q5 (c)
    # v,pi = gridW.Q5_c()
    # print("Q5 (c)\n", v, '\n',pi)

    Jack = env.JacksCarRental()
    Value,Policy = Jack.Q6_a()

    # py = Jack.make_likelihood(0)

    # print(np.sum(py[20]))
