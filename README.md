# reinforcement-learning-notes
My implementations of RL algorithms.

# Notes
### Value Iteration and Policy Iteration
* The **state-value function** (V(s)) measures the desirability of states, while the **action-value function** (Q(s,a)) measures the desirability of taking specific actions in those states.
* In Value Iteration, (V(s)) is effectively derived from (Q(s,a)) via the max operator. After convergence, the same ends up being true for Policy Iteration, even though it has a separate Policy Evaluation step. They’re both “hunting” for the same optimal Bellman equation, just taking different routes.
* In my experiments, with a discount factor (gamma = 1), Value Iteration converges almost immediately, while Policy Iteration struggles. This seems to be because VI doesn’t try to fully solve the environment at each step. It performs a single sweep over states (almost like a “lazy PI”). Policy Iteration, on the other hand, runs into instability during Policy Evaluation as values grow large and become harder to stabilise.
* That said, this behaviour may be specific to small, discrete environments, and also an issue of my iterative implementation, as most references suggest PI should converge in fewer iterations than VI.

### SARSA and Q-Learning
* SARSA is an on-policy TD method since it updates the Q-table based on the action taken, i.e new Q-value for a state-action pair is based on the action a' taken when in state s', while Q-Learning is an off-policy method since updates to the Q-table are independent of the learned policy, i.e new Q-value for a state-action pair is obtained from s' regardless of the action taken.
* State Action Rewards nextState nextAction (SARSA).
* Given that Q-Learning updates the Q-table using the maximum Q-value of the next state, it tends to have much higher Q-values than SARSA. (obviously :-)). It seems that in Q-Learning, the Q-table gets updated using the state-value of the s', drawing some parallels to Value Iteration.
* Across all the algorithms, the learned state-values are highest closer to the goal state.
