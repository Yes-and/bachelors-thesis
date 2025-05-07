import concurrent.futures
import datetime
import multiprocessing
import os
import shutil

from src.buffers import MultiprocessingExperienceMemory
from src.logger import setup_logger
from src.networks import REINFORCE

from training.simulate_game import REINFORCE_play_game
from training.train_network import REINFORCE_train

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim



class GlobalVariables:
    # State representation
    STATE_SIZE = 34
    ACTION_SIZE = 18

    # Hyperparameters
    LR = 1e-3  # Learning rate
    N_GAMES_TRAINING = 128 # Essentially the batch size
    EPISODES = 100000  # Training episodes

    # Others
    NUM_GAMES_PARALLEL = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./models/REINFORCE-policy-net.pth"

    # Logging
    SAVE_MODEL_FREQUENCY = 5000



if __name__ == "__main__":
    g = GlobalVariables()
    multiprocessing.set_start_method("spawn", force=True)

    # Delete the old model
    if os.path.exists(g.MODEL_PATH):
        os.remove(g.MODEL_PATH)
        print(f"Deleted existing file: {g.MODEL_PATH}")
    
    # Create random model, save it and delete from memory
    policy_net = REINFORCE(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    torch.save(policy_net.state_dict(), g.MODEL_PATH)
    optimizer = optim.Adam(policy_net.parameters(), lr=g.LR)

    # Initialize Logging
    logger, log_queue = setup_logger()
    writer = SummaryWriter()

    # Log hyperparameters
    logger.info("Training a REINFORCE agent.")
    logger.info(f"LR: {g.LR}")
    logger.info(f"N_GAMES_TRAINING: {g.N_GAMES_TRAINING}")
    logger.info(f"NUM GAMES PARALLEL: {g.NUM_GAMES_PARALLEL}")

    # Shared experience memory optimized for multiprocessing
    safe_exp_buffer = MultiprocessingExperienceMemory()
    game_counter = 1

    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=g.NUM_GAMES_PARALLEL) as executor:
        next_train, next_save_model = g.N_GAMES_TRAINING, g.SAVE_MODEL_FREQUENCY

        while True:
            # Start multiple games in parallel
            futures = [executor.submit(REINFORCE_play_game, g) for _ in range(g.NUM_GAMES_PARALLEL)]

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
                        logger.info(f"Game {game_counter} completed.")
                        game_counter += 1
                    else:
                        logger.warning("Worker returned no experiences.")
                except concurrent.futures.TimeoutError:
                    logger.error("Game execution timed out.")
                except Exception as e:
                    logger.error(f"Exception in worker: {str(e)}")

            # Train after collecting a full batch
            if game_counter >= next_train:
                logger.info(f"Collected {len(safe_exp_buffer)} experiences. Training now...")
                states, actions, rewards, _, _ = safe_exp_buffer.get_batch()
                loss, entropy = REINFORCE_train(
                    g=g,
                    policy_net=policy_net,
                    optimizer=optimizer,
                    states=states,
                    actions=actions,
                    rewards=rewards
                )
                writer.add_scalar("loss", loss, game_counter)
                writer.add_scalar("entropy", entropy, game_counter)
                logger.info("Policy network for REINFORCE has been fitted. Clearing the buffer...")
                safe_exp_buffer.clear()
                next_train += g.N_GAMES_TRAINING

            # Save model periodically
            if game_counter >= next_save_model:
                logger.info("Saving policy network model for REINFORCE.")
                timestamp = str(datetime.datetime.now())[:19].replace(":", "-")
                shutil.copy("./models/REINFORCE-policy-net.pth", f"./models/REINFORCE-model-{timestamp}.pth")
                logger.info("Policy network for REINFORCE agent has been saved.")
                next_save_model += g.SAVE_MODEL_FREQUENCY

            if game_counter >= g.EPISODES:
                break

    writer.close()

    # Cleanup temp model file
    os.remove(g.MODEL_PATH)
