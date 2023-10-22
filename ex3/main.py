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

    # Q5 (a)
    gridW = env.Gridworld5x5()
    print("Q5 (a)\n", gridW.Q5_a())
    # Q5 (b)
    vb,pb = gridW.Q5_b()
    print("Q5 (b)\n", vb, '\n',pb)
    # Q5 (c)
    vc,pc = gridW.Q5_c()
    print("Q5 (c)\n", vc, '\n',pc)

    # Q6
    Jack = env.JacksCarRental()
    Value,Policy = Jack.Q6()  # For swtiching between Q6a and Q6b, need to comment  out
                            # the different cost function at line 377, in _calculate_cost()

    # px, py = Jack._open_to_close(1)
    # print(np.sum(px, axis =1))
    # print(px[25])
