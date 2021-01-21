# Self-Improving Chatbots based on Reinforcement Learning

Code accompanying the paper:

Elena Ricciardelli, Debmalya Biswas. Self-improving Chatbots based on Reinforcement Learnin. In proceedings of the 4th Multidisciplinary Conference on Reinforcement Learning and Decision Making (RLDM), Montreal, 2019
https://www.researchgate.net/publication/333203489_Self-improving_Chatbots_based_on_Reinforcement_Learning

-------------------------------------------------------------------

We present a Reinforcement Learning (RL) model for self-improving chatbots, specifically targeting FAQ-type chatbots. The model is not aimed at building a dialog system from scratch, but to leverage data from user conversations to improve chatbot performance. At the core of our approach is a score model, which is trained to score chatbot utterance-response tuples based on user feedback. The scores predicted by this model are used as rewards for the RL agent. Policy learning takes place offline, thanks to an user simulator which is fed with utterances from the FAQ-database. Policy learning is implemented using a Deep Q-Network (DQN) agent with epsilon-greedy exploration.

The DQN agent is initially trained offline in a warm-up phase on the NLU. The score model is also trained offline with the data  from real user conversations. In the RL loop, the user state (user utterance) is provided by the user simulator, the action (chatbot response) is provided by the DQN agent and the reward is provided by the score model. Each tuple (state, action, reward) feeds the experience replay buffer, which is used to re-train the DQN after n_episodes episodes,
The NLU can be either rasa or watson. The model architecture is shown in archi_v2.png.

Future improvement of the code will include  FAQ with slots, thus going beyond simple question-answer chatbots. 

-----------------------------------------
Installation
-----------------------------------------

Dependencies:
- rasa with tf pipeline [not needed if using watson ]
- watson_developer_cloud [not needed if using rasa ]
- tensorflow
- tensorflow-hub
- pandas
- numpy

-------------------------------------------
Input files:
------------------------------------------------
- logs of  full conversation
- logs in [utterance, response, confidence, feedback] format for the number of intents chosen, to train score model (it can be created in load_data())
- test set for a bunch of conversation with golden intent (manually labelled) [optional]
- intent-response mapping
- NLU training set
  if using rasa:  deep_dialog/nlu/rasa_config/GStraining.json
  if using watson: it's in the watson workspace
- If BoW embeddings are used insred of the universal sentence encoder, the glove file should be downloaded and copied in: deep_dialog/nlu

-------------------------------------------
Running the code
-----------------------------------------------
set configuration in utils.py
Run as:
python run.py

