import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            n_rows=5,
            n_cols=5,
            start_state=(0, 0),
            goal_state=(4, 4),
            trap_states=(),
            step_reward=-1.0,
            goal_reward=10.0,
            trap_reward=-10.0,
            render_mode=None
    ):
        super().__init__()

        # Grid definition
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start_state = start_state
        self.goal_state = goal_state
        self.trap_states = list(trap_states)
        self.render_mode = render_mode

        # ✅ Rewards (FIXED)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward

        # Gym spaces (DataCamp-style)
        self.observation_space = spaces.Discrete(n_rows * n_cols)
        self.action_space = spaces.Discrete(4)

        self.nS = self.observation_space.n
        self.nA = self.action_space.n

        self.state = None

        # Transition matrix
        self.P = self._build_P()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _to_state_id(self, state):
        r, c = state
        return r * self.n_cols + c

    def _from_state_id(self, sid):
        return divmod(sid, self.n_cols)

    def _in_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _is_terminal(self, state):
        return state == self.goal_state or state in self.trap_states

    def _move(self, state, action):
        r, c = state
        moves = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        if not self._in_bounds(nr, nc):
            return state
        return (nr, nc)

    # --------------------------------------------------
    # Transition model P[s][a]
    # --------------------------------------------------
    def _build_P(self):
        P = {}

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                state = (r, c)
                s = self._to_state_id(state)
                P[s] = {}

                for a in range(self.nA):
                    if self._is_terminal(state):
                        P[s][a] = [(1.0, s, 0.0, True)]
                        continue

                    next_state = self._move(state, a)
                    ns = self._to_state_id(next_state)

                    if next_state == self.goal_state:
                        reward = self.goal_reward
                        done = True
                    elif next_state in self.trap_states:
                        reward = self.trap_reward
                        done = True
                    else:
                        reward = self.step_reward
                        done = False

                    P[s][a] = [(1.0, ns, reward, done)]

        return P

    # --------------------------------------------------
    # Gym API
    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        return self._to_state_id(self.state), {}

    def step(self, action):
        s = self._to_state_id(self.state)
        _, ns, reward, done = self.P[s][action][0]
        self.state = self._from_state_id(ns)
        return ns, reward, done, False, {}

    def render(self):
        grid = np.zeros((self.n_rows, self.n_cols, 3), dtype=np.uint8)

        # Fill with white
        grid[:, :] = [255, 255, 255]

        # Goal = green
        gr, gc = self.goal_state
        grid[gr, gc] = [0, 255, 0]

        # Traps = red
        for r, c in self.trap_states:
            grid[r, c] = [255, 0, 0]

        # Agent = blue
        r, c = self.state
        grid[r, c] = [0, 0, 255]

        if self.render_mode == "human":
            # Print ASCII for human
            ascii_grid = np.full((self.n_rows, self.n_cols), ".", dtype=str)
            for rr, cc in self.trap_states:
                ascii_grid[rr, cc] = "X"
            ascii_grid[gr, gc] = "G"
            ascii_grid[r, c] = "A"
            for row in ascii_grid:
                print(" ".join(row))
            print()

        elif self.render_mode == "rgb_array":
            # Return the NumPy image array
            return grid


    @property
    def unwrapped(self):
        return self
