from utils import *
#import watson_developer_cloud
import random
from deep_dialog.agents.agent import Agent
from deep_dialog.nlu.nlu import set_nlu
from deep_dialog.nlu.q_table import Q_table as Q_tab
from deep_dialog.nlu.watson import  watson
import copy
import pickle
import os
import sys
from learning_scores.embeddings import *
from learning_scores.score_model import *
import os
from statsmodels.stats.proportion import proportion_confint
import json
from episodes import *
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_path=None #use model file if you want to start from pre-trained model. 

######################################################
# load configuration
######################################################
params = parameters()

#######################################################
# load logs data --> used by the simulator
######################################################
df, dict_intents, top_intents = load_data(params, max_intents = 6)

#######################################################
# Load NLU agent - rasa or watson 
######################################################
NLU = set_nlu(params, 'rasa', top_intents)

# return dict with mapping of sets of states/actions to index
mapping = get_mapping(NLU, df) #defined in utils.py

#######################################################
# Load user Simulator (defined in utils.py)
######################################################
simulator = Simulator(df, multiply=4)


#######################################################
# Init Agent (defined in deep_dialog/agents/agent.py)
######################################################
warm_start = 1
agent = Agent(NLU, params, mapping, warm_start, model_path=None)


#######################################################
#  WARM-UP phase (warmup_run defined in episodes.py)
######################################################
if model_path is None:
        
        # train Q in warm-up phase --> reward from Watson CI. Set use_Q to True if you want the NLU resp to be taken from Q table (pickle file already created)
        agent, Q_table = warmup_run(agent, simulator, params, mapping, use_Q=False, verbose=False)

        # test DQN accuracy by cfr DQN with NLU results
        testing(agent, mapping, Q_table, verbose=False)
        print('... END of WARM-UP PHASE')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


#######################################################
#  Train score model
######################################################

# load score mdoel configuration (in utils.py)
params_model = config_score_model()
    
# load dat to train score model (utils.py)
x, y, emb_u, emb_r = score_data()

#init score model (in learning_scores/score_model.py)
model = score_model(params_model)

print('Training score model...')
Final_train_acc, _ = model.fit(x,y, emb_u=emb_u, emb_r=emb_r, path_to_model = "./trained_model/model_final.ckpt")
    
print('Train Accuracy of score model:', Final_train_acc)
print('----------------------------------------------------')


#######################################################
#  Run episodes
######################################################
print('---------- Running RL episodes -----------')
results = episodes_run(agent, NLU, simulator, model,  params, mapping)

print('End of episodes !')
print()          

# write summary of results to file
results.to_csv('summary.csv', index=False)






