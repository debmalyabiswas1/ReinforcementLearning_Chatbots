import random, copy, json
import pickle
import numpy as np
import sys
import copy
from deep_dialog.qlearning import DQN

class Agent(object):
  """ 
  Class defining the RL agent behaviour

  Methods:
  ----------------------------------------------------
  register_experience_replay_tuple(state, action, reward, state_tplus1, episode_over, buffer_size=10000, flush=False):
       register state-action-reward tuple in the buffer
  run_policy(representation, epoch, return_q=False):
       run epsilon-greedy policy
  load_trained_DQN(path):
       load pre-trained model
  train(batch_size=1, num_batches=100, num_iter =1000, mini_batches = False)"
       train DQN, can be done in mini-batches or on the whole sample
  """
      
  def __init__(self, nlu, params, mapping, warm_start, model_path=None):

        self.nlu = nlu

        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.hidden_size = params['dqn_hidden_size']
        self.num_epochs = params['num_episodes']/params['train_freq']
        self.epsilon_epoch_f = params['epsilon_epoch_f']
        self.epsilon_f = params['epsilon_f']
        self.prob_no_intent = params['prob_no_intent']
        self.nlu_type = params['nlu']
        self.buffer_size = params['buffer_size']
        
        # epsilon decay low: from large exploration to large explotation
        self.eps_decay = (1-self.epsilon_f/self.epsilon)/float(self.epsilon_epoch_f)
        #self.eps_decay = (1-self.epsilon_f/self.epsilon)/float(self.num_epochs)

        self.map_action2index = mapping['action2index']
        self.map_state2index = mapping['state2index']
        self.map_index2state = mapping['index2state']
        self.map_index2action = mapping['index2action']

        self.possible_states = self.map_index2state
        self.user_act_cardinality = len(self.map_index2state.keys()) #len(user_act_set.keys())
        self.state_dimension = len(self.map_index2state)
        self.feasible_actions = self.map_index2action
        self.agent_act_cardinality = len(self.map_index2action.keys()) #len(agent_act_set.keys())
        self.num_actions = len(self.feasible_actions)

        self.warm_start = warm_start

        #self.state = StateTracker() # to be implemented for slots

        self.experience_replay_pool = []

        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)

        self.clone_dqn = copy.deepcopy(self.dqn)

        self.model_path = model_path
        
        if self.model_path is not None:
              
              self.dqn.model = copy.deepcopy(self.load_trained_DQN(self.model_path))
              self.clone_dqn = copy.deepcopy(self.dqn)

        return


  def state_to_action(self, state):
        """ DQN: Input state, output action """

        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


  def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, buffer_size=10000, flush=False):
        """ Register feedback from the environment, to be stored as future training data """


        # user action one-hot representation
        user_act_rep = np.zeros((1,self.user_act_cardinality))
        user_act_rep[0, self.map_state2index[s_t]] = 1.0

        # agent action one-hot representation
        agent_act_rep = self.map_action2index[a_t]
        
        # next user action one-hot representation
        next_user_act_rep = np.zeros((1,self.user_act_cardinality))
        next_user_act_rep[0, self.map_state2index[s_tplus1]] = 1.0

        reward_t = reward

        training_example = (user_act_rep, agent_act_rep, reward_t, next_user_act_rep, episode_over)


        if flush:
            
            #print('flushing...', np.size(self.experience_replay_pool), len(self.experience_replay_pool),
            #          max(0, np.size(self.experience_replay_pool)-buffer_size))
            
            to_delete = max(0, len(self.experience_replay_pool) - self.buffer_size)
            print('Deleting first %d examples from buffer' % to_delete)
            del self.experience_replay_pool[:to_delete]
            
                      
        self.experience_replay_pool.append(training_example)
        self.example = training_example
        

        return

  def run_policy(self, representation, epoch, return_q=False):
        """ epsilon-greedy policy """

        if epoch is not None:
              
              epsilon = self.epsilon*(1-self.eps_decay*epoch)
              r = random.random()
              
        else: #use rule policy
              print('Running rule policy')
              return self.dqn.predict(representation, {'gamma': self.gamma}, predict_model=True, return_q=return_q)
          
        if r < epsilon:
            
            ran_action = random.randint(0, self.num_actions-1)
            r2 = random.random()
            if r2 < self.prob_no_intent: #increase prob of having No intent detected
                ran_action =   self.num_actions-1

            #return random.randint(0, self.num_actions-1)
            print('------  random action ----------', r, r2, epsilon)
            return ran_action
        else:

            return self.dqn.predict(representation, {'gamma': self.gamma}, predict_model=True)

  
  def rule_policy(self, user):
        """ NLU Policy  """

        return self.nlu.predict(user)
  
            
  def load_trained_DQN(self, path):
        """ Load trained model from pickle file """
        
        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']

        print('DQN loaded from file')
        
        return model
        
  def train_ontable(self, batch_size=1, num_batches=100):

        for iter_batch in range(1000):
            
            self.cur_bellman_err = 0
            batch = self.experience_replay_pool
            batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
            self.cur_bellman_err += batch_struct['cost']['total_cost']

  def train(self, batch_size=1, num_batches=100, num_iter =1000, mini_batches = False):
        """ Train DQN with experience replay """

        if mini_batches:
        # Mini-batches
            for iter_batch in range(num_batches):
                self.cur_bellman_err = 0
                for iter in range(int(len(self.experience_replay_pool)/(batch_size))):
                    batch = [random.choice(self.experience_replay_pool) for i in range(batch_size)]
                    batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
                    self.cur_bellman_err += batch_struct['cost']['total_cost']

                    rep = np.zeros((1,self.user_act_cardinality))
 
        else:
            # Train on the whole sample
           
            check_points = [int(num_iter*a*0.1) for a in range(1,10)]

            for iter_batch in range(num_iter):

                self.cur_bellman_err = 0
                batch = self.experience_replay_pool
                
                batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
                
                self.cur_bellman_err += batch_struct['cost']['total_cost']


