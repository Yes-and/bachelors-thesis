class GameExperienceMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.intermediate_rewards = []
        self.end_turns = None
        self.end_reward = None

class SingleExperienceMemory:
    def __init__(self, state, action, reward, turn, end_turns, intermediate_reward=None, next_state=None, done=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.turn = turn
        self.end_turns = end_turns
        self.intermediate_reward = intermediate_reward
        self.next_state = next_state
        self.done = done
