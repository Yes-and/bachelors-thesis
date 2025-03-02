# RL application for card game Dominion

Information about this repository
* All of the used packages should be in requirements.txt file
* Folder *bin* is unused
* Folder *src* contains all of the important modifications of pyminion so that it can run
* File *rl_agent_play.py* is currently the main testing scenario
* TensorBoard is used for monitoring runs

How can I run the main script?
1. Create a venv (python3 -m venv venv)
2. Activate it (source venv/bin/activate)
3. Install requirements (pip install -r requirements.txt)
4. Run the main script (rl_agent_play.py)

What are the results of the latest run?
* The model seems to favor decisions that provide negative VPs
* The loss value is oscillating (and not decreasing)
* Q-value is exploding
I will have to try tuning hyperparameters or altering architecture to achieve better results.
