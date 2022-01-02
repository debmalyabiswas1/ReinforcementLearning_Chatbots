import pandas as pd
import numpy as np


class evaluate(object):
    """ evaluate RL performances on test set extracted from the full conversations and manually labeled """
    
    def __init__(self, map_state2index, map_index2action, file='GS_test_set.csv'):
        """ read test set """
        
        DF = pd.read_csv(file)
        
        self.utt_test = list(DF['utterance'])
        self.gold_resp = list(DF['gold response'])
        
        self.map_state2index = map_state2index
        self.map_index2action = map_index2action
        self.size_test = len(self.utt_test)

    def fit(self, agent):
        """ run evaluation"""
        
        success = 0
        num_utt = 0
        print('#### Start evaluationa #######')
        for i, utterance in enumerate(self.utt_test):

            if utterance in self.map_state2index:
                
                repr = get_representation(utterance, self.map_state2index)

                index_action = agent.run_policy(repr, None)
                
                if index_action in self.map_index2action:

                    num_utt += 1
                    action = self.map_index2action[index_action]
                    gold = self.gold_resp[i]
                    if action == gold:
                        success += 1
                    print('---------------------------------')
                    print('utterance:', utterance, i)
                    print('action:', action)
                    print('golden response:', gold)
                    print('success:', success)

 
        print('#### End of evaluation #######')
        success_rate =  float(success)/float(num_utt)       
        #print('Success Rate:', success_rate, success, num_utt)

        return success_rate 

def get_representation(utterance, map_state2index):
    """ return one-hot representation of the given sentence """
    
    size = len(map_state2index)
    index = map_state2index[utterance]
    rep = np.zeros((1,size))
    rep[0, index] = 1.0

    return rep
