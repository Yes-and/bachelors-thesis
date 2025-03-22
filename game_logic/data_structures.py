class GameExperienceMemory:
    def __init__(self):
        self.state = None
        self.action = None
        self.turn = None
        self.end_turns = None
        self.end_reward = None

    def store(self, state, action, turn, end_turns, end_reward):
        """Store a single transition."""
        self.state = state
        self.action = action
        self.turn = turn
        self.end_turns = end_turns
        self.end_reward = end_reward
