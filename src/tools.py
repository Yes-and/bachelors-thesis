from collections import deque

import torch



def save_replay_buffer_torch(replay_buffer, filename="unnamed_replay_buffer.pth"):
    torch.save(list(replay_buffer), filename)
    print(f"Replay buffer saved to {filename}")

def load_replay_buffer_torch(filename="replay_buffer.pth", buffer_size=100000):
    try:
        data = torch.load(filename)
        replay_buffer = deque(data, maxlen=buffer_size)
        print(f"Replay buffer loaded from {filename}, containing {len(replay_buffer)} experiences")
        return replay_buffer
    except FileNotFoundError:
        print("No existing replay buffer found. Starting fresh.")
        return deque(maxlen=buffer_size)
