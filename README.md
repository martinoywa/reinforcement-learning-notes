# reinforcement-learning-notes
My implementations of RL algorithms.

# Notes
### Value Iteration and Policy Iteration
* The **state-value function** (V(s)) measures the desirability of states, while the **action-value function** (Q(s,a)) measures the desirability of taking specific actions in those states.
* In Value Iteration, (V(s)) is effectively derived from (Q(s,a)) via the max operator. After convergence, the same ends up being true for Policy Iteration, even though it has a separate Policy Evaluation step. They’re both “hunting” for the same optimal Bellman equation, just taking different routes.
* In my experiments, with a discount factor (gamma = 1), Value Iteration converges almost immediately, while Policy Iteration struggles. This seems to be because VI doesn’t try to fully solve the environment at each step. It performs a single sweep over states (almost like a “lazy PI”). Policy Iteration, on the other hand, runs into instability during Policy Evaluation as values grow large and become harder to stabilise.
* That said, this behaviour may be specific to small, discrete environments, and also an issue of my iterative implementation, as most references suggest PI should converge in fewer iterations than VI.
