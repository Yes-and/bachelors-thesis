from multiprocessing import Manager

class MultiprocessingExperienceMemory:
    def __init__(self):
        manager = Manager()
        self.states = manager.list()
        self.actions = manager.list()
        self.rewards = manager.list()
        self.next_states = manager.list()
        self.dones = manager.list()

    def store_batch(self, new_experience):
        """Store an entire batch of experiences at once to reduce locking overhead."""
        self.states.append(new_experience.state)
        self.actions.append(new_experience.action)
        self.rewards.append(new_experience.reward)
        self.next_states.append(new_experience.next_state)
        self.dones.append(new_experience.done)

    def clear(self):
        """Clear all stored experiences."""
        self.states[:] = []
        self.actions[:] = []
        self.rewards[:] = []
        self.next_states[:] = []
        self.dones[:] = []

    def get_batch(self):
        """Return normal lists before training."""
        return (
            list(self.states),
            list(self.actions),
            list(self.rewards),
            list(self.next_states),
            list(self.dones)
        )

    def __len__(self):
        return len(self.states)
