# AI Path Detection  in Drone Using ML

This project demonstrates **AI-powered drone obstacle avoidance** using machine learning. A Decision Tree Classifier predicts drone navigation decisions based on real-time sensor inputs.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Code Explanation](#code-explanation)
3. [Installation](#installation)
4. [How It Works](#how-it-works)
5. [Output](#output)
6. [Usage](#usage)

---


## 🎯 Overview

This script simulates an intelligent drone that:
- **Detects obstacles** using three sensors (left, front, right)
- **Makes real-time decisions** using a trained AI model
- **Navigates autonomously** while avoiding obstacles
- **Visualizes the complete path** taken during the flight

---

## 📝 Code Explanation

### 1. **Model Training** (Lines 1-20)

```python
from sklearn.tree import DecisionTreeClassifier

X = [
    [1,0,0],  # obstacle left
    [0,1,0],  # obstacle front
    [0,0,1],  # obstacle right
    [0,0,0]   # clear path
]

y = [0,1,2,3]  # Actions: 0=Right, 1=Stop, 2=Left, 3=Forward

model = DecisionTreeClassifier()
model.fit(X, y)
```

**What it does:**
- Creates a **Decision Tree** machine learning model
- Trains it on 4 sensor scenarios and their corresponding actions
- The model learns to map sensor inputs → navigation decisions

**Sensor Inputs:**
- `[1,0,0]` = Obstacle detected on the **left**
- `[0,1,0]` = Obstacle detected **in front**
- `[0,0,1]` = Obstacle detected on the **right**
- `[0,0,0]` = **Clear path** ahead

**AI Actions:**
- `0` = Move Right (➜)
- `1` = Stop (⛔)
- `2` = Move Left (⬅)
- `3` = Move Forward (⬆)

---

### 2. **Initial Position** (Lines 22-27)

```python
x = 0
y_pos = 0

x_points = [x]
y_points = [y_pos]
```

**What it does:**
- Sets the drone's starting position at coordinates **(0, 0)**
- `x_points` stores all X-coordinates visited
- `y_points` stores all Y-coordinates visited
- Both lists are used for path visualization

---

### 3. **Simulation Loop** (Lines 29-65)

```python
for step in range(10):
    # Automatic Sensor Generation
    left = random.randint(0,1)
    front = random.randint(0,1)
    right = random.randint(0,1)

    # AI Prediction
    prediction = model.predict([[left, front, right]])

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

    # Store Path
    x_points.append(x)
    y_points.append(y_pos)
```

**What it does:**
- Runs for 10 time steps
- **Generates random sensor values** (0 or 1) to simulate obstacle detection
- **Makes an AI prediction** based on the trained model
- **Executes the decision** and updates drone position:
  - Right: `x += 1`
  - Stop: No position change
  - Left: `x -= 1`
  - Forward: `y_pos += 1`
- **Records the path** for visualization

---

### 4. **Path Visualization** (Lines 67-106)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

# Plot the complete path
plt.plot(x_points, y_points, marker='o', linewidth=3, label='Drone Path')

# Add step numbers at each position
for i in range(len(x_points)):
    plt.text(x_points[i], y_points[i], str(i), fontsize=12)

# Mark start and end points
plt.scatter(x_points[0], y_points[0], s=200, label='START')
plt.scatter(x_points[-1], y_points[-1], s=200, label='END')

plt.title("AI Drone Path Detection")
plt.xlabel("X Movement")
plt.ylabel("Y Movement")
plt.grid(True)
plt.legend()
plt.show()
```

**What it does:**
- Creates a matplotlib visualization
- Plots the entire drone path as a line with markers
- Labels each step with numbers (0-10)
- Marks the starting position
- Marks the ending position
- Displays a grid and legend

---

## 💻 Installation

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn
```

### Requirements
- Python 3.6+
- numpy
- matplotlib
- scikit-learn

---

## 🚀 How It Works

### Step-by-Step Process

1. **Training Phase**: The model learns 4 rules:
   - Obstacle on left → Move right
   - Obstacle in front → Stop
   - Obstacle on right → Move left
   - Clear path → Move forward

2. **Simulation Phase**: For 10 steps:
   - Generate random sensor readings
   - Ask the AI model: "What should I do?"
   - Execute the decision
   - Record the position

3. **Visualization Phase**: Plot the complete path showing:
   - Drone trajectory
   - All waypoints numbered
   - Start and end positions

---

## 📊 Output

### Console Output Example

```
----- AI DRONE PATH DETECTION -----

STEP 1
Sensor Values: 1 0 0
AI Decision: MOVE RIGHT ➜

STEP 2
Sensor Values: 0 1 0
AI Decision: STOP ⛔

STEP 3
Sensor Values: 0 0 1
AI Decision: MOVE LEFT ⬅

STEP 4
Sensor Values: 0 0 0
AI Decision: MOVE FORWARD ⬆

STEP 5
Sensor Values: 1 0 0
AI Decision: MOVE RIGHT ➜

... (continues for 10 steps)
```

### Graph Output

A 2D scatter plot showing:
- **Drone path** (blue line with circles)
- **Step numbers** (0-10) at each point
- **START point** (green marker)
- **END point** (orange marker)
- **Grid** for reference

---

## 🎮 Usage

### Run the Script

```bash
python drone_obstacle_avoidance.py
```

### Output
1. Prints step-by-step sensor readings and AI decisions to console
2. Displays an interactive matplotlib graph
3. Shows complete drone path with waypoints

---

## 🔧 Technical Details

| Component | Details |
|-----------|---------|
| **Algorithm** | Decision Tree Classifier |
| **Input Features** | 3 sensors (left, front, right) |
| **Output Classes** | 4 actions (right, stop, left, forward) |
| **Simulation Steps** | 10 timesteps |
| **Sensor Values** | Binary (0=No obstacle, 1=Obstacle) |

---

## 📌 Key Concepts

### Why Decision Trees?
- **Fast prediction** for real-time drone control
- **Interpretable** decisions based on sensor data
- **Efficient** for small datasets
- **Suitable** for autonomous navigation

### Real-World Applications
- Drone obstacle avoidance systems
- Autonomous vehicle navigation
- Robot path planning
- Mobile obstacle detection

---

## 📖 References

- [scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [Matplotlib Visualization](https://matplotlib.org/)
- [NumPy Array Operations](https://numpy.org/)

---

## 🤝 Contributing

Feel free to contribute improvements to this project!

