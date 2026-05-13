import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import random

# -----------------------------------
# TRAIN AI MODEL
# -----------------------------------

# Sensor Inputs
# [Left, Front, Right]

X = [
    [1,0,0],  # obstacle left
    [0,1,0],  # obstacle front
    [0,0,1],  # obstacle right
    [0,0,0]   # clear path
]

# AI Actions
# 0 = Move Right
# 1 = Stop
# 2 = Move Left
# 3 = Move Forward

y = [0,1,2,3]

model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------------
# START POSITION
# -----------------------------------

x = 0
y_pos = 0

x_points = [x]
y_points = [y_pos]

# -----------------------------------
# AI PATH DETECTION
# -----------------------------------

print("\n----- AI DRONE PATH DETECTION -----\n")

for step in range(10):

    # Automatic Sensor Generation
    left = random.randint(0,1)
    front = random.randint(0,1)
    right = random.randint(0,1)

    print("STEP", step+1)
    print("Sensor Values:",
          left,
          front,
          right)

    # AI Prediction
    prediction = model.predict(
        [[left, front, right]]
    )

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
# PATH VISUALIZATION
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
plt.title("AI Drone Path Detection")

plt.xlabel("X Movement")

plt.ylabel("Y Movement")

plt.grid(True)

plt.legend()

plt.show()
