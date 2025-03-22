from multiprocessing import Manager

class MultiprocessingExperienceMemory:
    def __init__(self):
        manager = Manager()
        self.states = manager.list()
        self.actions = manager.list()
        self.turns = manager.list()
        self.end_turns = manager.list()
        self.end_rewards = manager.list()

    def store_batch(self, new_experience):
        """Store an entire batch of experiences at once to reduce locking overhead."""
        self.states.append(new_experience.state)
        self.actions.append(new_experience.action)
        self.turns.append(new_experience.turn)
        self.end_turns.append(new_experience.end_turns)
        self.end_rewards.append(new_experience.end_reward)

    def clear(self):
        """Clear all stored experiences."""
        self.states[:] = []
        self.actions[:] = []
        self.turns[:] = []
        self.end_turns[:] = []
        self.end_rewards[:] = []

    def get_batch(self):
        """Return normal lists before training."""
        return (
            list(self.states),
            list(self.actions),
            list(self.turns),
            list(self.end_turns),
            list(self.end_rewards)
        )

    def __len__(self):
        return len(self.states)
