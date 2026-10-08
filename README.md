# Grid-World MDP Solver

A Python implementation of two classic dynamic-programming algorithms for solving Markov Decision Processes (MDPs): **Value Iteration** and **Policy Iteration**. The agent moves on a 2D grid with walls, terminal states and stochastic movement, and the program computes the optimal value of every cell and the optimal policy.

## Features

- Value Iteration and Policy Iteration, both built on the Bellman equation
- Stochastic transition model with diagonal "slips"
- Grid and rewards loaded from a NumPy `.npz` file
- Heat-map image of the state values
- Image and terminal printout of the optimal policy as arrows
- Convergence graph for Policy Iteration

## The Model

**States.** Every free cell of the grid is a state. Cells whose absolute reward is greater than 0.1 are terminal states; all other free cells are active states.

**Actions.** `UP`, `DOWN`, `LEFT`, `RIGHT`. An action is available only if the target cell is not a wall and is inside the grid.

**Transitions.** Movement is not deterministic:

| Outcome | Probability |
|---|---|
| The agent moves in the chosen direction | 0.8 |
| The agent slips to one of the two diagonal cells in that direction | 0.1 each |

If a diagonal cell is blocked, its probability is added to the chosen direction.

**Value of an action.**

```
Q(s, a) = Σ P(s' | s, a) · [ R(s') + γ · U(s') ]
```

**Parameters.**

| Parameter | Value |
|---|---|
| Discount factor γ | 0.9 |
| Success probability | 0.8 |
| Convergence threshold ε | 0.01 |
| Stop condition | max change < ε · (1 − γ) / γ |

## Algorithms

### Value Iteration

Starts with zero values (terminal states hold their reward) and repeatedly updates every active state to the value of its best action, until the largest change in a sweep falls below the stop condition or 100 iterations are reached. The optimal policy is then extracted by choosing the best action in each state.

### Policy Iteration

1. **Initialization** – each state gets the first legal action in the order `UP`, `DOWN`, `RIGHT`, `LEFT`.
2. **Policy evaluation** – the values of the current policy are computed iteratively, using the same stop condition.
3. **Policy improvement** – each state switches to a better action if one exists.
4. Steps 2–3 repeat until the policy no longer changes.

## Requirements

- Python 3
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## Usage

```bash
python MDP.py <input_file.npz> <algorithm>
```

`<algorithm>` is either `ValueIteration` or `PolicyIteration`.

```bash
python MDP.py input1_2026a.npz ValueIteration
python MDP.py input1_2026a.npz PolicyIteration
```

## Input Format

A `.npz` file containing two matrices of the same shape:

| Key | Content |
|---|---|
| `states` | `1` for a free cell, `0` for a wall |
| `rewards` | The reward of each cell |

## Output

All images are saved to the `Output/` folder, which is created automatically.

| File | Content |
|---|---|
| `<Algorithm>_Values_RonLapushner.jpg` | Heat map of the state values |
| `<Algorithm>_Policy_RonLapushner.jpg` | Table of the optimal policy |
| `PolicyIteration_RonLapushner.jpg` | Number of evaluation sweeps in each policy-iteration round (Policy Iteration only) |

The policy is also printed to the terminal. Value Iteration additionally prints the maximum change of each iteration.

Illustration of the printout format:

```
Optimal Policy:
-----------------
| → | → | → | o |
-----------------
| ↑ | x | ↑ | o |
-----------------
| ↑ | ← | ↑ | ← |
-----------------
```

| Symbol | Meaning |
|---|---|
| `↑ ↓ ← →` | Best action in the cell |
| `o` | Terminal state |
| `x` | Wall |

## Project Structure

```
MDP.py      # The MDP class and the command-line entry point
Output/     # Generated images
```
