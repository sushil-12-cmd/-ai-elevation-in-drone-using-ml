import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

X = [
    [10],
    [20],
    [30],
    [40],
    [50],
    [60],
    [70],
    [80]
]

y = [0, 0, 1, 1, 1, 2, 2, 2]

model = DecisionTreeClassifier()
model.fit(X, y)

distance = np.arange(0, 20, 1)

terrain_elevation = np.array([
    10, 15, 20, 25, 30,
    40, 50, 60, 70, 80,
    75, 65, 55, 45, 35,
    25, 20, 15, 10, 5
])

controlled_elevation = []

drone_speed = 12
safe_altitude = 15

print("\nAI DRONE ELEVATION CONTROL SYSTEM\n")

for height in terrain_elevation:

    prediction = model.predict([[height]])

    if prediction[0] == 0:
        decision = "INCREASE ALTITUDE"
        new_height = height + safe_altitude

    elif prediction[0] == 1:
        decision = "MAINTAIN ALTITUDE"
        new_height = height + 5

    else:
        decision = "DECREASE ALTITUDE"
        new_height = height - 5

    controlled_elevation.append(new_height)

    print("Terrain Elevation :", height, "m")
    print("Drone Decision    :", decision)
    print("Drone Altitude    :", new_height, "m")
    print("Drone Speed       :", drone_speed, "m/s")
    print("--------------------------------------")

highest_index = np.argmax(terrain_elevation)

plt.figure(figsize=(14, 7))

plt.plot(
    distance,
    terrain_elevation,
    marker='o',
    linewidth=3,
    label='Terrain Elevation'
)

plt.plot(
    distance,
    controlled_elevation,
    marker='s',
    linestyle='--',
    linewidth=3,
    label='AI Drone Path'
)

plt.scatter(
    distance[highest_index],
    terrain_elevation[highest_index],
    s=250,
    label='Highest Terrain Point'
)

plt.title("AI Drone Elevation Control System", fontsize=18)

plt.xlabel("Distance", fontsize=14)

plt.ylabel("Elevation Height (m)", fontsize=14)

plt.grid(True)

plt.legend()

plt.show()
