# RL application for card game Dominion

Information about this repository
* All of the used packages should be in requirements.txt file
* Folder *bin* is unused
* Folder *game_logic* contains updated game logic so that the RL agent can run
  * This includes the modified game and bot from *pyminion*
* Folder *training* includes all the functions that are used during the training
  * Game simulation and network training for different RL agents is located here
* File *train_DQN* is used for training the DQN. It has been simplified from the previous iterations.
* File *train_A2C* is used for training the A2C (actor-critic). It has also been simplified.
* File *train_REINFORCE* is used for training the REINFORCE agent.
* File *test_network* is used for debugging of the decisions done by the network
* TensorBoard is used for monitoring runs

How can I run the main script?
1. Create a venv (python3 -m venv venv)
2. Activate it (source venv/bin/activate)
3. Install requirements (pip install -r requirements.txt)
4. Run the main script

Results so far:
* DQN is not learning
* A2C is learning, but very slowly
* REINFORCE learns the fastest and can beat BigMoney bot 
