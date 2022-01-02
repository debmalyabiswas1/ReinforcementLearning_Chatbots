import numpy as np
import pickle

class Q_table(object):

    def __init__(self, mapping) :

        self.map_state2index = mapping['state2index']
        self.map_index2state = mapping['index2state']
        self.map_action2index = mapping['action2index']
        self.map_index2action = mapping['index2action']

        self.learning_rate = 0.01
        self.epsilon = 0.2

        self.num_states = len(self.map_state2index)
        self.num_actions = len(self.map_action2index)
        print('Q table size [states, actions]:', self.num_states, self.num_actions)

        self.Q = np.zeros([self.num_states, self.num_actions])


    def add(self, value, i_s, i_a):

        self.Q[i_s,i_a] = value

        return

    def predict(self, i_s):

        Q_row = self.Q[i_s, :]

        return np.argmax(Q_row), np.amax(Q_row)

    def save_Q(self, pfile = 'Q.pck'):

        with open(pfile, 'wb') as f:
            pickle.dump(self.Q, f)

        return

    def load_Q(self, pfile='Q.pck'):

        with open(pfile, 'rb') as f:
            self.Q = pickle.load(f)

        return

    def learn(self, index_state, index_action, reward):

        current_q = self.Q[index_state, index_action]
        # using Bellman Optimality Equation to update q function
        new_q = float(reward) #+ self.discount_factor * max(self.q_table[next_state])

        self.Q[index_state, index_action] += self.learning_rate * (new_q - current_q)

        return

    def greedy_policy(self, index_state):

        if random.random() < self.epsilon:
            # random exploration
            print('eps policy:', random.randint(0, self.num_actions - 1), self.num_actions)
            return random.randint(0, self.num_actions - 1)
        else:
            # exploitation
            print('index_state:', index_state)
            index_action, _ = self.predict(index_state)
            return index_action
