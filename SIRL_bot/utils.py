import pandas as pd
import random
import numpy as np
import copy
from learning_scores.embeddings import *

def parameters():
    """ configuration for RL agent """
    
    config = {}
    config['log_file'] = 'GS_logs.csv' #file with conversation
    config['nlu'] = 'rasa' #either watson or rasa
    config['num_episodes_warm'] = 100 # should be similar to number of conversations 
    config['train_freq_warm'] = 100 # if equal to num_episodes_war --> 1 training epoch, which is enough for warmp-up
    config['num_episodes'] = 100 # total number of episodes 
    config['train_freq'] =  10 #30 number of episodes per training epochs
    config['epsilon'] = 0.2 #initial
    config['epsilon_f'] = 0.05 #epsilon after epsilon_epoch_f epochs
    config['epsilon_epoch_f'] = 20  
    config['dqn_hidden_size'] = 60 #paramter of DQN
    config['gamma'] = 0.9 #paramter of DQN 
    config['prob_no_intent'] = 0.5 #probability of giving 'No intent detected" in random policy
    config['buffer_size'] = 10000 # max size of experience replay buffer 
    return config

def config_score_model():
    """ configuration for score model """
    
    params = {}
    params['epochs'] = 500
    params['lr'] = 1.e-4
    params['l2_reg'] = 0.1
    params['print_freq'] = 100

    return params

def watson_config():
    """ set configuration to connect to Watson workspace ID if NLU=Watson """
    
    username = ''
    password =''
    WID = ''

    return


#def load_data(file='GS_logs.csv', max_intents = 7):
def load_data(params, max_intents = 7 ):    
    """ Load dataset with conversation already transformed in conventient format: utterance, response, confidence, feedback, along with intents counter dict and list of top intents """

    print('------------- Loading conversations ---------------')

    file = params['log_file']
    df = pd.read_csv(file)
    
    df.dropna(subset=['confidence'], how='all', inplace = True)
    df.dropna(subset=['utterance'], how='all', inplace = True)
    df.dropna(subset=['response'], how='all', inplace = True)

    print('lenght of original DataFrame:', len(df.index))
    
    df_top_intents, dict_intent, top_intents = get_top_intents(df, max_num_intents = max_intents)
    print('Size of  top intents dataframe:', len(df_top_intents.index))
    
    # remove feedback intent, as it does not have response
    df_top_intents = df_top_intents[df_top_intents['intent'] != 'Feedback']

    print()
    
    return df_top_intents, dict_intent, top_intents

def get_df_feedback(df):
    """ transform original dataset of covnewrsation into convenient format with feedback """
    
    indices = []
    feedbacks = []
    count_feedback = 0

    for index, data in df.iterrows():

        utt_feedback =['no', 'yes']
        utt = data['utterance']
        intent = data['intent']
        confidence = data['confidence']

        utt = utt.lower()

        indices.append(index)

        is_feedback = any([utt.startswith(f) for f in utt_feedback])

        if intent.lower() == 'feedback':

            feedback = 0 if utt.startswith('yes') else 1

            feedbacks.append({'utterance': previous['utterance'], 'response': previous['response'],
            'feedback':feedback, 'confidence':previous['confidence'], 'intent':previous['intent']})

            count_feedback += 1

        previous = data   

    print('Number of feedbacks:', count_feedback)
    f = [item['feedback'] for item in feedbacks]
    print('positive:',f.count(0))
    print('negative:',f.count(1))

    return pd.DataFrame(feedbacks)



def get_top_intents(df, max_num_intents = 90, remove_fallback=True):
    """ This function selects the max_num_intents intents more frequently triggered by the users and return dataframe with these intents only"""

    df_feedback = get_df_feedback(df)

    intents = list(df_feedback['intent'])
    unique_intents = list(set(intents))
    print('number of intents triggered:', len(unique_intents))

    count_intent = []

    for intent in unique_intents:

        dict_intent = {}
        dict_intent['intent'] = intent
        dict_intent['counts'] = intents.count(intent)
        count_intent.append(dict_intent)

    counts = np.array([dic['counts'] for dic in count_intent])
    isort = np.argsort(counts)[::-1]

    count_intent_sorted = np.array(count_intent)[isort]

    top_intents = [dictionary['intent'] for dictionary in count_intent_sorted[0:max_num_intents]]

    print('top_intents:', top_intents)

    if 'No intent detected' in top_intents  and remove_fallback:
        top_intents.remove('No intent detected')

    df_top_intents = df_feedback.loc[df_feedback['intent'].isin(top_intents)]
 

    return df_top_intents, count_intent, top_intents

def score_data(file='./learning_scores/GS_short_corrected.csv', thre_augm = 0.9):
    """ Load and transform (embeddings) data to train score model """
    
    ### Prepare Data for training score_model
    dialog_short = pd.read_csv(file)

    scores_inv = np.array(dialog_short['feedback'])
    
    #revert scale such that: 0=bad; 1=good
    scores = np.where( scores_inv == 0, 1, 0) 
    confidence = np.array(dialog_short['confidence'])

    # CL used only for data augmentation
    scores_comb = list(0.5*np.add(scores, confidence)) 
    
    scores = list(scores)
    
    utt= list(dialog_short['utterance'])
    resp = list(dialog_short['response'])
    intents = list(dialog_short['intent'])

    # data augmentation for high CL responses
    intents_u = list(set(intents))

    ind = np.flatnonzero( np.array([np.array(scores_comb) > thre_augm ]))

    bad_intent = 'No intent detected'
    bad_resp = "Sorry, I can't help you with that. Please direct your question to the GS Local Support (GeneralServicesLocalSupportLausanne)"

    for ind0 in ind:

        utt.append(utt[ind0])
        intents.append(bad_intent)
        resp.append(bad_resp)
        scores.append(0)

        
    EMB = embeddings([], embed_par='tf')
 
    emb_utt = EMB.fit(utt)
    emb_resp = EMB.fit(resp)

    emb_s = EMB.embed_size

    #### Delete NaN
    ind = np.argwhere(np.isnan(emb_utt[:,0]))

    emb_utt = np.delete(emb_utt, ind, axis = 0)
    emb_resp = np.delete(emb_resp, ind, axis = 0)
    scores = np.delete(scores, ind, axis = 0 )

    m = np.shape(emb_utt)[0]

    concat = np.concatenate((emb_utt, emb_resp))

    x = concat.reshape([2, m, emb_s])
    y = scores.astype(np.float32)
    
    return x, y, emb_utt, emb_resp

class Simulator(object):

    def __init__(self,  conv_dict, multiply=4):
        """ initialize parent sample of conversation and random generator """
        
        self.conv_dict = conv_dict

        all_utt = list(conv_dict['utterance'])
        feedback = list(conv_dict['feedback'])
        
        ## define parent sample of conversations: self.utt such that extraction
        ## of utterance triggering resp  with negative feedback has higher probability
        ## probability ratio negative:positive defined by multiply
        wrongs = []
        good = []
        utt = []
        
        for i in range(len(all_utt)):
            if feedback[i] == 0:
                utt.append(all_utt[i])
            else:
                 for k in range(multiply):
                     utt.append(all_utt[i])

        self.utt = utt             
        self.N = len(self.utt)
        
        print()
        print('---------------------------------------------------------------------------')
        print('Parent sample of conversations in Simulator contains %d questions'%self.N)

        ## init random generator
        seed = 45896
        random.seed(seed)

        self.iterator = conv_dict.iterrows()

        return

    def run_random(self):
        """ Returns random action """
        
        r = random.randint(0, self.N-1)       
        return self.utt[r]

    def sequential(self, reset=False):
        """ return action in sequential order """
        
        if reset:

            self.iterator = None
            self.iterator = self.conv_dict.iterrows()
            print('resetting...',self.iterator)

        try:
            return  next(self.iterator)[1]
        except StopIteration:
            return None
        



def state_to_dict(df):

    utterances = list(df['utterance'])
    unique_utterances = list(set(utterances))
    print('# utterances in final DF:', len(utterances), len(list(set(utterances))))

    count = 0
    map_index2state = {}
    map_state2index = {}

    for utt in unique_utterances:

        map_index2state[count] = utt
        map_state2index[utt] = count

        count += 1

    return map_index2state, map_state2index

def get_mapping(NLU, df):
    """ return dictionary mapping set of states (utterances) to indices and viceversa, and set of actions (responses) to indices and viceversa """
    
    map_index2action, map_action2index = NLU.actions_to_dict()
    map_index2state, map_state2index = state_to_dict(df)

    mapping = {}
    mapping['index2action'] = map_index2action
    mapping['action2index'] = map_action2index
    mapping['index2state'] = map_index2state
    mapping['state2index'] = map_state2index

    return mapping
