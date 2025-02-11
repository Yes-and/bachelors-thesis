# RL application for card game Dominion

Information about this repository
* All of the used packages should be in requirements.txt file
* Folder *bin* is unused
* Folder *src* contains all of the important modifications of pyminion so that it can run
* File *short_test.py* is currently the main testing scenario
* TensorBoard is used for monitoring runs

How can I run the main script?
1. Create a venv (python3 -m venv venv)
2. Activate it (source venv/bin/activate)
3. Install requirements (pip install -r requirements.txt)
4. Run the main script (short_test.py)

What are the results of the latest run?
* It seems, that the agent is not learning very well
* This could be due to two reasons:
  * The reward is too sparse (I will add more reward scenarios)
  * The replay buffer is too small (Currently at 5k experiences, 10k would be optimal)
  * Game representation is not optimal (I will normalize the state information values)
  * ... and some other causes ...
