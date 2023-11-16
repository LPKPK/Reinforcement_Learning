import numpy as np
import matplotlib.pyplot as plt

STARTING = 0.8 # the value of the starting line
FINISHING = 0.4 # the value of the finishing line

def build_track_a(save_map=False):
    track = np.ones(shape=(32, 17))
    track[14:, 0] = 0
    track[22:, 1] = 0
    track[-3:, 2] = 0
    track[:4, 0] = 0
    track[:3, 1] = 0
    track[0, 2] = 0
    track[6:, -8:] = 0 

    track[6, 9] = 1

    track[:6, -1] = FINISHING
    track[-1, 3:9] = STARTING 
    if save_map:
        with open('track_a.npy', 'wb') as f:
            np.save(f, track)
    return track


def build_track_b(save_map=False):
    track = np.ones(shape=(30, 32))

    for i in range(14):
        track[:(-3 - i), i] = 0
    
    track[3:7, 11] = 1
    track[2:8, 12] = 1
    track[1:9, 13] = 1
   
    track[0, 14:16] = 0
    track[-17:, -9:] = 0
    track[12, -8:] = 0
    track[11, -6:] = 0
    track[10, -5:] = 0
    track[9, -2:] = 0

    track[-1] = np.where(track[-1] == 0, 0, STARTING)
    track[:, -1] = np.where(track[:, -1] == 0, 0, FINISHING)
    if save_map:
        with open('track_b.npy', 'wb') as f:
            np.save(f, track)
    return track

if __name__ == "__main__":
    print(build_track_a())