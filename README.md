# reinforcement-learning-notes
My implementations of RL algorithms.

# Notes
### V(s) vs Q(s,a) in Value Iteration and Policy Iteration
- State-value Function (V(s)) measures the disirability of states, while Action-value Function (Q(s,a)) the disirability of actions within those states.
- V(s) will contain the Q-value for derived from Q(s,a) especially in Value Iteration. This will also be true in Policy Iteration after convergence even though it has a separate Policy  Evaluation function. This because they're both  "hunting" for the same Optimal Bellman Equation solution regardless of route.
- In my experiments, with a Discount Faction (gamma=1), Value Iteration converges immediately, while Policy Iteration struggles. This is because Value Iteration doesn't care about solving the full environment. It just does a single sweep of the states and moves one. Policy Iteration on the other hand, struggles in its Policy Evaluation step because values become large and too hard to stabilise.
- Note that the above may only be true in small discrete environments because according to available text, PI iterates faster than VI, or the iterative implementatio is just slow.