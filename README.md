# AI Drone Elevation and Obstacle Avoidance Using ML

## Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import random

# -----------------------------------
# TRAIN AI MODEL - OBSTACLE AVOIDANCE
# -----------------------------------

# Sensor Inputs for Obstacle Avoidance
# [Left, Front, Right]

X_obstacle = [
    [1,0,0],  # obstacle left
    [0,1,0],  # obstacle front
    [0,0,1],  # obstacle right
    [0,0,0]   # clear path
]

# AI Actions for Obstacle Avoidance
# 0 = Move Right
# 1 = Stop
# 2 = Move Left
# 3 = Move Forward

y_obstacle = [0,1,2,3]

model_obstacle = DecisionTreeClassifier()
model_obstacle.fit(X_obstacle, y_obstacle)

# -----------------------------------
# TRAIN AI MODEL - ELEVATION CONTROL
# -----------------------------------

X_elevation = [
    [10],
    [20],
    [30],
    [40],
    [50],
    [60],
    [70],
    [80]
]

y_elevation = [0, 0, 1, 1, 1, 2, 2, 2]

model_elevation = DecisionTreeClassifier()
model_elevation.fit(X_elevation, y_elevation)

# -----------------------------------
# TERRAIN AND DISTANCE DATA
# -----------------------------------

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

# -----------------------------------
# START POSITION FOR OBSTACLE AVOIDANCE
# -----------------------------------

x = 0
y_pos = 0

x_points = [x]
y_points = [y_pos]

# -----------------------------------
# AI PATH DETECTION - OBSTACLE AVOIDANCE
# -----------------------------------

print("\n----- AI DRONE OBSTACLE AVOIDANCE PATH DETECTION -----\n")

for step in range(10):

    # Automatic Sensor Generation
    left = random.randint(0,1)
    front = random.randint(0,1)
    right = random.randint(0,1)

    print("STEP", step+1)
    print("Sensor Values:", left, front, right)

    # AI Prediction
    prediction = model_obstacle.predict([[left, front, right]])

    # AI Navigation Logic

    if prediction[0] == 0:
        decision = "MOVE RIGHT ➜"
        x += 1

    elif prediction[0] == 1:
        decision = "STOP ⛔"

    elif prediction[0] == 2:
        decision = "MOVE LEFT ⬅"
        x -= 1

    else:
        decision = "MOVE FORWARD ⬆"
        y_pos += 1

    print("AI Decision:", decision)
    print()

    # Store Path
    x_points.append(x)
    y_points.append(y_pos)

# -----------------------------------
# AI DRONE ELEVATION CONTROL SYSTEM
# -----------------------------------

print("\n----- AI DRONE ELEVATION CONTROL SYSTEM -----\n")

for height in terrain_elevation:

    prediction = model_elevation.predict([[height]])

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

# -----------------------------------
# PATH VISUALIZATION - OBSTACLE AVOIDANCE
# -----------------------------------

plt.figure(figsize=(10,6))

# Plot Drone Path
plt.plot(
    x_points,
    y_points,
    marker='o',
    linewidth=3,
    label='Drone Path'
)

# Add Step Numbers
for i in range(len(x_points)):
    plt.text(
        x_points[i],
        y_points[i],
        str(i),
        fontsize=12
    )

# Start Point
plt.scatter(
    x_points[0],
    y_points[0],
    s=200,
    label='START'
)

# End Point
plt.scatter(
    x_points[-1],
    y_points[-1],
    s=200,
    label='END'
)

# Labels
plt.title("AI Drone Path Detection - Obstacle Avoidance")
plt.xlabel("X Movement")
plt.ylabel("Y Movement")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------------
# ELEVATION CONTROL VISUALIZATION
# -----------------------------------

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
```

## Installation

```bash
pip install numpy matplotlib scikit-learn
```

## Requirements
- Python 3.6+
- numpy
- matplotlib
- scikit-learn

## Features

- **Obstacle Avoidance**: AI-powered drone navigation using sensor inputs (left, front, right)
- **Elevation Control**: Autonomous altitude adjustment based on terrain elevation
- **Decision Tree Classifier**: ML model for real-time decision making
- **Path Visualization**: 2D visualization of drone path during obstacle avoidance
- **Elevation Visualization**: Terrain and controlled elevation comparison
- **Real-time Decisions**: Autonomous decision making at each step

## Output

The script generates:
1. Console output showing step-by-step decisions for obstacle avoidance
2. Console output showing elevation control decisions for terrain navigation
3. 2D scatter plot of drone path avoiding obstacles
4. 2D line plot comparing terrain elevation vs AI-controlled drone elevation
