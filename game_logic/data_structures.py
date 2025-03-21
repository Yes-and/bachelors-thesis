class GameExperienceMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.state_values = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, state_value, log_prob, reward, done):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.state_values.append(state_value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)
