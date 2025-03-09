from multiprocessing import Manager

class ExperienceMemory:
    def __init__(self):
        manager = Manager()
        self.states = manager.list()
        self.actions = manager.list()
        self.log_probs = manager.list()
        self.rewards = manager.list()
        self.dones = manager.list()

    def store(self, state, action, log_prob, reward, done):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def store_batch(self, new_experiences):
        """Store an entire batch of experiences at once to reduce locking overhead."""
        self.states.extend(new_experiences.states)
        self.actions.extend(new_experiences.actions)
        self.log_probs.extend(new_experiences.log_probs)
        self.rewards.extend(new_experiences.rewards)
        self.dones.extend(new_experiences.dones)

    def clear(self):
        """Clear all stored experiences."""
        self.states[:] = []
        self.actions[:] = []
        self.log_probs[:] = []
        self.rewards[:] = []
        self.dones[:] = []

    def __len__(self):
        return len(self.states)
