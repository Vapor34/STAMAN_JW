# This is the document of dexterous robot hand grasping #


This repository focuses on advanced robotic manipulation, combining heuristic optimization for grasp planning with robust force-based control strategies. The project aims to generate optimal pre-grasp poses and execute stable grasps in complex environments.

## 🚀 What have done

### 1. Pre-grasp Pose Space Generation (Simulated Annealing)

We utilize the **Simulated Annealing (SA)** algorithm to explore the high-dimensional pose space of the robotic end-effector.

* **Heuristic Search:** Efficiently identifies optimal pre-grasp candidates by minimizing a multi-objective cost function.
* **Pose Optimization:** Balances reachability, collision avoidance, and gripper alignment.

### 2. Force Control

Implementation of active force control to manage interactions between the gripper and the object.

* **Admittance Control:** * Implementing a mass-spring-damper dynamics model to allow the robot to react compliantly to external forces.

### 3. Grasp Evaluation
* **Grasp Scoring System:** * A comprehensive evaluation suite to rank generated poses.
* Metrics will include Force Closure and GWS (Grasp Wrench Space) analysis.

## 🛠 Planned Modules (Roadmap)

We are actively expanding the framework to include the following features:
* **RL**
* **Improvement of Planning Strategy**







## Shortcuts

![Grasp pos1](/readme_doc/pos1.png)

![Grasp pos1](/readme_doc/pos3.png)

![Grasp pos1](/readme_doc/pos4.png)

![Grasp pos1](/readme_doc/pos2.png)

![Grasp pos1](/readme_doc/pos5.png)

![Grasp pos1](/readme_doc/eval1.png)

![Grasp pos1](/readme_doc/eval2.png)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git

# Install dependencies
# create a virtual env

conda create --name env_name pyhton=3.12
conda activate

# in venv
conda install pip
pip install -r requirements.txt 

```

## 💻 Usage

1. **Simulation:**
Run the main.py
```bash
python main.py
```






