import concurrent.futures
import datetime
import multiprocessing
import os
import shutil

from src.buffers import MultiprocessingExperienceMemory
from src.logger import setup_logger
from src.networks import DQN

from training.simulate_game import DQN_play_game
from training.train_network import DQN_train

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim



class GlobalVariables:
    # State representation
    STATE_SIZE = 34
    ACTION_SIZE = 18

    # Hyperparameters
    GAMMA = 0.99
    LR = 1e-4  # Learning rate
    N_GAMES_TRAINING = 128 # Essentially the batch size
    EPISODES = 100000  # Training episodes

    # Epsilon decay
    EPSILON_START = 1 # Epsilon decay schedule recommended by ChatGPT
    EPSILON_FINAL = 0.1
    EPSILON_DECAY_LAST_GAME = 80000
    EPSILON = EPSILON_START

    # Other parameters
    NUM_GAMES_PARALLEL = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./models/temp-DQN-net.pth"

    # Saving the model
    SAVE_MODEL_FREQUENCY = 5000



if __name__ == "__main__":
    g = GlobalVariables()
    multiprocessing.set_start_method("spawn", force=True)

    # Delete the old model
    if os.path.exists(g.MODEL_PATH):
        os.remove(g.MODEL_PATH)
        print(f"Deleted existing file: {g.MODEL_PATH}")
    
    # Create random model, save it and delete from memory
    q_net = DQN(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    torch.save(q_net.state_dict(), g.MODEL_PATH)
    optimizer = optim.Adam(q_net.parameters(), lr=g.LR)

    # Initialize Logging
    logger, log_queue = setup_logger()
    writer = SummaryWriter()

    # Log hyperparameters
    logger.info("Training DQN agent")
    logger.info(f"GAMMA: {g.GAMMA}")
    logger.info(f"LR: {g.LR}")
    logger.info(f"N_GAMES_TRAINING: {g.N_GAMES_TRAINING}")
    logger.info(f"EPSILON_START: {g.EPSILON_START}")
    logger.info(f"EPSILON_FINAL: {g.EPSILON_FINAL}")
    logger.info(f"EPSILON_DECAY_LAST_FRAME: {g.EPSILON_DECAY_LAST_GAME}")
    logger.info(f"NUM GAMES PARALLEL: {g.NUM_GAMES_PARALLEL}")

    # Shared experience memory optimized for multiprocessing
    safe_exp_buffer = MultiprocessingExperienceMemory()
    game_counter = 1

    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=g.NUM_GAMES_PARALLEL) as executor:
        next_train, next_save_model = g.N_GAMES_TRAINING, g.SAVE_MODEL_FREQUENCY

        while True:
            # Start multiple games in parallel
            futures = [executor.submit(DQN_play_game, g) for _ in range(g.NUM_GAMES_PARALLEL)]

            # Wait for all workers to complete before continuing
            completed, _ = concurrent.futures.wait(futures)

            for future in completed:
                try:
                    new_experience, player_vp, enemy_vp = future.result(timeout=10)
                    if new_experience is not None:
                        safe_exp_buffer.store_batch(new_experience)  # Store experiences in batch

                        writer.add_scalar("victory_points_player", player_vp, game_counter)
                        writer.add_scalar("victory_points_enemy", enemy_vp, game_counter)
                        writer.add_scalar("player_turns", new_experience.end_turns, game_counter)
                        writer.add_scalar("player_reward", new_experience.reward, game_counter)
                        writer.add_scalar("epsilon", g.EPSILON, game_counter)

                        g.EPSILON = max(
                            g.EPSILON_FINAL, g.EPSILON_START - game_counter / g.EPSILON_DECAY_LAST_GAME
                        ) # Update EPSILON decay
                        game_counter += 1
                        logger.info(f"Game {game_counter} completed.")
                    else:
                        logger.warning("Worker returned no experiences.")
                except concurrent.futures.TimeoutError:
                    logger.error("Game execution timed out.")
                except Exception as e:
                    logger.error(f"Exception in worker: {str(e)}")

            # Train after collecting a full batch
            if game_counter >= next_train:
                logger.info(f"Collected {len(safe_exp_buffer)} experiences. Training now...")
                states, actions, rewards, next_states, dones = safe_exp_buffer.get_batch()
                loss, mean_q_val, std_q_val = DQN_train(
                    g=g,
                    q_net=q_net,
                    optimizer=optimizer,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                    target_net=None
                )
                writer.add_scalar("loss", loss, game_counter)
                writer.add_scalar("mean_q_val", mean_q_val, game_counter)
                writer.add_scalar("std_q_val", std_q_val, game_counter)
                logger.info("DQN has been fitted. Clearing the buffer...")
                safe_exp_buffer.clear()
                next_train += g.N_GAMES_TRAINING

            # Save model periodically
            if game_counter >= next_save_model:
                logger.info("Saving DQN model.")
                timestamp = str(datetime.datetime.now())[:19].replace(":", "-")
                shutil.copy("./models/temp-DQN-net.pth", f"./models/DQN-model-{timestamp}.pth")
                logger.info("DQN has been saved.")
                next_save_model += g.SAVE_MODEL_FREQUENCY

            if game_counter >= g.EPISODES:
                break

    writer.close()

    # Cleanup temp model file
    os.remove(g.MODEL_PATH)
