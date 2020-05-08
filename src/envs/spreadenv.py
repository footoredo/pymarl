from smac.env import MultiAgentEnv
import numpy as np


class SpreadEnv(MultiAgentEnv):
    def __init__(self, n_agents, seed=None, state_last_action=False, obs_last_action=False, obs_use_simple_scheme=False):
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios

        self._seed = seed

        self.scenario = scenarios.load("simple_spread.py").SpreadScenario(n_agents, seed)
        self.world = self.scenario.make_world()
        self.env = MultiAgentEnv(self.world, self.scenario.reset_world, self.scenario.reward, self.scenario.observation)
        self.env.discrete_action_input = True
        self.n_agents = n_agents
        self.n_actions = self.world.dim_p * 2 + 1
        self.episode_limit = 25

        self.state_last_action = state_last_action
        self.obs_last_action = obs_last_action
        self.obs_use_simple_scheme = obs_use_simple_scheme

        self._episode_steps = 0

        self.last_action = [np.zeros(self.n_actions) for _ in range(self.n_agents)]

        super(SpreadEnv, self).__init__()

    def step(self, actions):
        self._episode_steps += 1
        obs_n, reward_n, done_n, info_n = self.env.step(actions)
        reward = reward_n[0]  # already shared reward
        terminated = all(done_n) or self._episode_steps >= self.episode_limit
        info = {}

        for agent_id, action in enumerate(actions):
            self.last_action[agent_id] = np.zeros(self.n_actions)
            self.last_action[agent_id][action] = 1.

        return reward, terminated, info

    def get_obs(self):
        obs_n = []
        for agent_id, agent in enumerate(self.env.agents):
            obs = self.get_obs_agent(agent_id)
            obs_n.append(obs)
        return obs_n

    def get_obs_agent(self, agent_id):
        n_agents = self.n_agents
        dim_p = self.world.dim_p
        dim_c = self.world.dim_c

        obs = self.env.get_obs(self.env.agents[agent_id])
        if self.obs_last_action:
            obs_a, obs_b = np.split(obs, [dim_p + dim_p + dim_p * n_agents])
            teammate_obs = np.split(obs_b, n_agents)
            assert teammate_obs[0].shape[0] == 1 + dim_p + dim_c
            new_teammate_obs = []
            for o in teammate_obs:
                no = np.concatenate([o, self.last_action[agent_id]])
                new_teammate_obs.append(no)
            obs = np.concatenate([obs_a] + new_teammate_obs)
        # print(agent_id, obs)

        # ts = np.zeros(self.episode_limit + 1)
        # ts[self._episode_steps] = 1
        ts = np.array([self._episode_steps / self.episode_limit])
        obs = np.concatenate([ts, obs])

        return obs

    def get_obs_size(self):
        n_agents = self.n_agents
        dim_p = self.world.dim_p
        dim_c = self.world.dim_c

        teammate_size = 1 + dim_p + dim_c
        if self.obs_last_action:
            teammate_size += self.n_actions

        # print(dim_p + dim_p + dim_p * n_agents + teammate_size * n_agents)

        return 1 + dim_p + dim_p + dim_p * n_agents + teammate_size * n_agents

    def get_scheme(self):
        """
        Returns the scheme of the observation and action
        """
        n_agents = self.n_agents
        dim_p = self.world.dim_p
        dim_c = self.world.dim_c

        teammate_size = 1 + dim_p + dim_c
        if self.obs_last_action:
            teammate_size += self.n_actions

        # self vel, self pos, landmark pos, other pos, comm
        if self.obs_use_simple_scheme:
            return [1 + dim_p + dim_p, (dim_p, n_agents), (teammate_size, n_agents)]
        else:
            return {
                "observation_pattern": [1 + dim_p + dim_p, (dim_p, "target"), (teammate_size, "agent")],
                "action_pattern": [1 + dim_p * 2],
                "objects": {
                    "target": n_agents,
                    "agent": n_agents
                }
            }

    def get_state(self):
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
            np.float32
        )
        # print("obs_concat shape", obs_concat.shape)
        return obs_concat

    def get_state_size(self):
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        self._episode_steps = 0
        self.env.reset()
        self.last_action = [np.zeros(self.n_actions) for _ in range(self.n_agents)]
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def render_frame(self):
        return self.env.render(mode='rgb_array')[0]

    def close(self):
        pass

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    def get_stats(self):
        stats = {
        }
        return stats

