import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

phone_tilt = np.pi/4
user_rotation = np.pi

vectors = [
    np.array([np.sin(phone_tilt)*np.cos(user_rotation), np.sin(phone_tilt)*np.sin(user_rotation), -np.cos(phone_tilt)]),
    np.array([np.sin(phone_tilt)*np.cos(user_rotation), np.sin(phone_tilt)*np.sin(user_rotation), -np.cos(phone_tilt)]),
    np.array([-np.sin(user_rotation), np.cos(user_rotation), 0])
]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

origin = np.array([0, 0, 0])
colors = ['r', 'g', 'b']

for vec, color in zip(vectors, colors):
    ax.quiver(*origin, *vec, color=color, length=1.0, normalize=True)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()