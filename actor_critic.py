import concurrent.futures
import datetime
import multiprocessing
import os
import shutil

from src.data_structures import MultiprocessingExperienceMemory
from src.logger import setup_logger
from src.actor_critic_model import ActorCritic

from training.simulate_game import play_game
from training.train_network import train_policy

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim



class GlobalVariables:
    # State representation
    STATE_SIZE = 34
    ACTION_SIZE = 18

    # Hyperparameters
    GAMMA = 0.99  # Discount factor
    LAMBDA = 0.95 # Bias-variance tradeoff factor
    LR = 1e-5  # Learning rate
    EPSILON_CLIP = 0.2  # PPO Clipping range
    EPOCHS = 10 # Number of tratates, actions, loning epochs per batch
    N_GAMES_TRAINING = 100  # N games for training
    BATCH_SIZE = 64 # Mini-batches for more efficient learning
    EPISODES = 5000  # Training episodes

    # Others
    NUM_GAMES_PARALLEL = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./models/temp-policy-net.pth"

    # Logging
    SAVE_MODEL_FREQUENCY = 1000
    LOG_FREQUENCY = 100



if __name__ == "__main__":
    g = GlobalVariables()
    multiprocessing.set_start_method("spawn", force=True)

    # Delete the old model
    if os.path.exists(g.MODEL_PATH):
        os.remove(g.MODEL_PATH)
        print(f"Deleted existing file: {g.MODEL_PATH}")
    
    # Create random model, save it and delete from memory
    policy_net = ActorCritic(g.STATE_SIZE, g.ACTION_SIZE).to(g.DEVICE)
    torch.save(policy_net.state_dict(), g.MODEL_PATH)
    optimizer = optim.Adam(policy_net.parameters(), lr=g.LR)

    # Initialize Logging
    logger, log_queue = setup_logger()
    writer = SummaryWriter()

    # Log hyperparameters
    logger.info("Training PPO while playing against BigMoney")
    logger.info(f"GAMMA: {g.GAMMA}")
    logger.info(f"LR: {g.LR}")
    logger.info(f"EPSILON CLIP: {g.EPSILON_CLIP}")
    logger.info(f"EPOCHS: {g.EPOCHS}")
    logger.info(f"BATCH SIZE: {g.N_GAMES_TRAINING}")
    logger.info(f"NUM GAMES PARALLEL: {g.NUM_GAMES_PARALLEL}")

    # Shared experience memory optimized for multiprocessing
    safe_exp_buffer = MultiprocessingExperienceMemory()
    game_counter = 1

    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=g.NUM_GAMES_PARALLEL) as executor:
        next_log, next_train, next_save_model = g.LOG_FREQUENCY, g.N_GAMES_TRAINING, g.SAVE_MODEL_FREQUENCY

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
                states, actions, rewards, log_probs, advantages, state_values, dones = batch
                loss, entropy, actor_loss, critic_loss = train_policy(
                    g=g,
                    policy_net=policy_net,
                    optimizer=optimizer,
                    states=states,
                    actions=actions,
                    log_probs=log_probs,
                    advantages=advantages,
                    state_values=state_values
                )
                writer.add_scalar("loss", loss, game_counter)
                writer.add_scalar("entropy", entropy, game_counter)
                writer.add_scalar("loss_actor", actor_loss, game_counter)
                writer.add_scalar("loss_critic", critic_loss, game_counter)
                logger.info("Policy network has been fitted. Clearing the buffer...")
                safe_exp_buffer.clear()
                next_train += g.N_GAMES_TRAINING

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
