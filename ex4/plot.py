import matplotlib.pyplot as plt
import numpy as np

def Q3_plot_value(ace_status, title):

    x1 = np.linspace(1, 10, num=10)
    y1 = np.linspace(12, 20, num=9)
    x1, y1 = np.meshgrid(x1, y1)
    z1 = ace_status[12:]

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax1.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='summer', linewidth=0, antialiased=False)
    fig1.colorbar(surf1, shrink=0.5, aspect=7)
    ax1.set_zlim(-1.0, 1.0)
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    plt.title(f"{title}")

def Q3_plot_policy(rPolicy, title):
    plt.figure()
    plt.imshow(np.flip(rPolicy[11:]), cmap='summer', extent=[1, 10, 11, 21])
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    plt.title(f"{title}")