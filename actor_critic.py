import concurrent.futures
import datetime
import multiprocessing
import os
import shutil

from src.data_structures import MultiprocessingExperienceMemory
from src.logger import setup_logger

from training.simulate_game import play_game
from training.train_network import train_policy

from tensorboardX import SummaryWriter
import torch



class GlobalVariables:
    # State representation
    STATE_SIZE = 38
    ACTION_SIZE = 18

    # Hyperparameters
    GAMMA = 0.99  # Discount factor
    LR = 1e-5  # Learning rate
    EPSILON_CLIP = 0.2  # PPO Clipping range
    EPOCHS = 5  # Number of tratates, actions, loning epochs per batch
    BATCH_SIZE = 32  # Mini-batch size for training
    EPISODES = 5000  # Training episodes

    # Others
    NUM_GAMES_PARALLEL = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./models/temp-policy-net.pth"

    # Logging
    SAVE_MODEL_FREQUENCY = 200
    LOG_FREQUENCY = 100



if __name__ == "__main__":
    g = GlobalVariables()
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize Logging
    logger, log_queue = setup_logger()
    writer = SummaryWriter()

    # Log hyperparameters
    logger.info("Training PPO while playing against BigMoney")
    logger.info(f"GAMMA: {g.GAMMA}")
    logger.info(f"LR: {g.LR}")
    logger.info(f"EPSILON CLIP: {g.EPSILON_CLIP}")
    logger.info(f"EPOCHS: {g.EPOCHS}")
    logger.info(f"BATCH SIZE: {g.BATCH_SIZE}")
    logger.info(f"NUM GAMES PARALLEL: {g.NUM_GAMES_PARALLEL}")

    # Shared experience memory optimized for multiprocessing
    safe_exp_buffer = MultiprocessingExperienceMemory()
    game_counter = 1

    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=g.NUM_GAMES_PARALLEL) as executor:
        next_log, next_train, next_save_model = g.LOG_FREQUENCY, g.BATCH_SIZE, g.SAVE_MODEL_FREQUENCY

        while True:
            # Start multiple games in parallel
            futures = [executor.submit(play_game, g) for _ in range(g.NUM_GAMES_PARALLEL)]

            # Wait for all workers to complete before continuing
            completed, _ = concurrent.futures.wait(futures)

            for future in completed:
                try:
                    new_experiences, player_vp, enemy_vp, turns = future.result(timeout=10)
                    if new_experiences is not None:
                        safe_exp_buffer.store_batch(new_experiences)  # Store experiences in batch
                        writer.add_scalar("victory_points_player", player_vp, game_counter)
                        writer.add_scalar("victory_points_enemy", enemy_vp, game_counter)
                        writer.add_scalar("turns_player", turns, game_counter)
                        logger.info(f"Game {game_counter} completed. Player VP: {player_vp}, Enemy VP: {enemy_vp}")
                        game_counter += 1
                    else:
                        logger.warning("Worker returned no experiences.")
                except concurrent.futures.TimeoutError:
                    logger.error("Game execution timed out.")
                except Exception as e:
                    logger.error(f"Exception in worker: {str(e)}")

            # Log at regular intervals
            if game_counter >= next_log:
                logger.info(f"{game_counter} games have been played.")
                next_log += g.LOG_FREQUENCY

            # Train after collecting a full batch
            if game_counter >= next_train:
                logger.info(f"Collected {len(safe_exp_buffer)} experiences. Training now...")
                batch = safe_exp_buffer.get_batch()
                states, actions, log_probs, rewards, dones = batch
                loss = train_policy(g, states, actions, log_probs, rewards)
                writer.add_scalar("loss", loss, game_counter)
                logger.info("Policy network has been fitted. Clearing the buffer...")
                safe_exp_buffer.clear()
                next_train += g.BATCH_SIZE

            # Save model periodically
            if game_counter >= next_save_model:
                logger.info("Saving policy network model.")
                timestamp = str(datetime.datetime.now())[:19].replace(":", "-")
                shutil.copy("./models/temp-policy-net.pth", f"./models/model-{timestamp}.pth")
                logger.info("Policy network has been saved.")
                next_save_model += g.SAVE_MODEL_FREQUENCY

            if game_counter >= g.EPISODES:
                break

    writer.close()

    # Cleanup temp model file
    os.remove(g.MODEL_PATH)
