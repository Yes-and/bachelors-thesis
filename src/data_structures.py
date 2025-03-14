from multiprocessing import Manager

class MultiprocessingExperienceMemory:
    def __init__(self):
        manager = Manager()
        self.states = manager.list()
        self.actions = manager.list()
        self.log_probs = manager.list()
        self.rewards = manager.list()
        self.dones = manager.list()

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

    def get_batch(self):
        """Return normal lists before training."""
        return (
            list(self.states),
            list(self.actions),
            list(self.log_probs),
            list(self.rewards),
            list(self.dones)
        )

    def __len__(self):
        return len(self.states)
