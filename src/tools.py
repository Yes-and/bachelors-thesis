from collections import deque

import numpy as np
import torch



def save_replay_buffer_torch(replay_buffer, filename="unnamed_replay_buffer.pth"):
    torch.save(list(replay_buffer), filename)
    print(f"Replay buffer saved to {filename}")

def load_replay_buffer_torch(filename="replay_buffer.pth", buffer_size=100000):
    try:
        data = torch.load(filename, weights_only=False)
        replay_buffer = deque(data, maxlen=buffer_size)
        print(f"Replay buffer loaded from {filename}, containing {len(replay_buffer)} experiences")
        return replay_buffer
    except FileNotFoundError:
        print("No existing replay buffer found. Starting fresh.")
        return deque(maxlen=buffer_size)

def sample_experience_batch(exp_buffer, batch_size):
    exp_list = list(exp_buffer)

    indices = np.random.choice(
        len(exp_list), batch_size, replace=False
    )
    states, actions, rewards, dones, next_states = \
        zip(*[exp_list[idx] for idx in indices])
    return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), \
            np.array(next_states)
