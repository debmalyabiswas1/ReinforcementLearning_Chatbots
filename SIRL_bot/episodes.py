from utils import *
from deep_dialog.agents.agent import Agent
from deep_dialog.nlu.q_table import Q_table as Q_tab
from evaluate import *
from statsmodels.stats.proportion import proportion_confint
import os
import copy
import pandas as pd
import pickle

def episodes_run(agent, NLU, simulator, model, params, mapping, output='epochs.csv', use_model = True, thre_no_intent=0.5):
    """ Run episodes and train DQN 
    Parameters:
    -------------------------------
    agent: instance of class Agent
    NLU: instance of class NLU
    simulator: instance of class Simulator
    model: instance of class score_model
    params: dictionary (RL agent configuration params)
    mapping: dict (maps from index to set of states/action and viceversa)
    output: string (optional)
    use_model: boolean (optional)
         if True use score model as rewards
         if False use interactive rewards (prompt the user)
    thre_no_intent: float (optional)     
    """

    ### Configure RL agent
    num_episodes = params['num_episodes']
    train_freq = params['train_freq']

    map_state2index = mapping['state2index']
    map_index2state = mapping['index2state']
    map_index2action = mapping['index2action']
    map_action2index = mapping['action2index']

    map_action2answer = pd.read_csv('intent_response.csv')
    num_states = len(map_index2state)


    ## Note for improvement: this computation can be done at an higher level and feed also the score_model
    emb_states = all_embeddings(map_index2state)
    emb_actions  = all_embeddings(map_index2action,  map_action = map_action2answer)

    # init evaluator
    eval = evaluate(map_state2index, map_index2action)
        
    sum_rewards = 0
    sum_rewards_bin = 0
    count_episodes = 0
    summary = []
    epoch = 0
    #episode_results = []
    #rewards_bin_episode = []
    #rewards_episode =[]

    flush = False

    utterance = simulator.run_random()
    

    for episode in range(num_episodes):

        # state_t: user question (plus slots if any: to be implemented)
        s_t = utterance
        print('----------------------------------------')
        print('Episode: %d' % episode)
        print('utterance: %s' % utterance)

        # get one-hot representation
        repr = get_representation(utterance, map_state2index)

        # get action index from agent running epsilon-greedy policy
        index_action = agent.run_policy(repr, epoch)

        # convert index to action
        action = map_index2action[index_action]

        print('intent:', action)

        if not use_model: #interactive rewards
            
            reward_int = -99
            reward_int = get_reward(utterance, action)
            reward = float(reward_int)
            
        else: # reward from score model 
            
            emb_u = emb_states[map_state2index[utterance]]
            emb_a = emb_actions[index_action]
        
            reward_model = get_reward_model([utterance], [action], [emb_u], [emb_a], model, map_action2answer)
            reward = reward_model

            ## The next branch is intended to overcome a bias in the data and it's probably not needed when more data
            ## will be included:
            ## the "No intent detected" gets biased low rewards (due to lack of positive examples in the logs)
            ## to give a more reliable reward, all other rewards are checked and the rewards is increased to 1  if no other intent gets a good enough reward
            ## otherwise the reward from model for "No intent detected"  is kept
            if action == 'No intent detected':

                other_utt = [utterance]*(len(map_action2index)-1)
                other_act = list(map_action2index.keys())
                other_act.remove('No intent detected')         
                emb_u = [emb_states[map_state2index[utterance]]] *(len(map_action2index)-1)
                emb_a = [emb_actions[map_action2index[action]] for action in other_act]                                            
                rs = get_reward_model(other_utt, other_act, emb_u, emb_a, model, map_action2answer)
 
                if np.array([rs >= thre_no_intent]).any():
                    reward = 1.
                    
        print('Reward: %d' % reward)
        reward_bin = 0 if reward<0.5 else 1
        
        s_t_plus1 = simulator.run_random()
        episode_over = True # this will change when slota are included
        
        sum_rewards = sum_rewards + float(reward)
        sum_rewards_bin = sum_rewards_bin + reward_bin
            
        count_episodes = count_episodes + 1
            
        # fill buffer and flush when flush=True (condition on flush set after training)          
        fill_buffer_all_actions(agent, s_t, action, reward, s_t_plus1, episode_over, flush=flush)

        if (episode > 0 and (episode+1) % train_freq == 0) :

            print('TRAINING DQN agent...')
            
            agent.clone_dqn = copy.deepcopy(agent.dqn)
            agent.train(4, 10, num_iter = 100)

            success_rate = eval.fit(agent) #success rate on test set
            re_bin_low, re_bin_up = proportion_confint(sum_rewards_bin, count_episodes) #CI based on binomial distr., alpha=0.05
            
            avg_score = float(sum_rewards)/float(count_episodes)
            avg_score_bin = float(sum_rewards_bin)/float(count_episodes)

            print('Average score for this epoch: %f, %f'%(avg_score, avg_score_bin))
            print('Success_rate on test set: %f' % success_rate)
            summary.append({'epoch': epoch, 'avg_score':avg_score, 'avg_score_bin': avg_score_bin, 'success_rate':success_rate})

            print('Summary:', summary)
            
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            
            save_model('./models/', agent, epoch)
            
            if avg_score > 0.7:
                flush = True
                
            sum_rewards = 0
            sum_rewards_bin = 0
            count_episodes = 0
            rewards_bin_episode = []
            rewards_episode = []
            
            epoch += 1

            
        utterance = s_t_plus1

        df_out = pd.DataFrame(summary, columns = ['epoch', 'avg_score', 'avg_score_bin', 'success_rate'] )
        
    return df_out 

def get_reward_model(utterances, actions, emb_u, emb_a, model, map_action2answer):
    """ score model inference to get reward """
    
    answers = [map_action2answer.loc[map_action2answer['intent'] == action]['response'].values[0] for action in actions]

    embed_size = np.size(emb_u[0])

    m = len(utterances)
    
    x = np.zeros([2,m,embed_size])

    for i in range(m):

        x[0,i,:] = np.array(emb_u[i])#[0:embed_size]
        x[1,i,:] = np.array(emb_a[i])#[0:embed_size]
 

    pred = model.predict(x, path_to_model = "./learning_scores/trained_model/model_final.ckpt")

    return pred


def get_representation(utterance, map_state2index):
    """ return one-hot representation of the given sentence """
    
    size = len(map_state2index)
    index = map_state2index[utterance]
    rep = np.zeros((1,size))
    rep[0, index] = 1.0

    return rep

def all_embeddings(map_index2sent, map_action=None):
    """ compute embeddings for all states or actions """
    
    indices, sentences = zip(*map_index2sent.items())

    if map_action is not None:
        
         answers = [map_action.loc[map_action['intent'] == action]['response'].values[0] for action in sentences]
         sentences = answers
    
    EMB = embeddings([], embed_par='tf')
    emb = EMB.fit(sentences)

    return dict(zip(indices, emb))


def warmup_run(agent, simulator, params, mapping, use_Q = True, verbose=True):
    """ Initial DQN training on NLU, usijg NLU confidence as reward """
    
 
    # allow use of Q-table, useful when using Watson to store results in a Q-table and avoid API calls
    Q_table = Q_tab(mapping)

    if use_Q:
        Q_table.load_Q()
        
    num_episodes = params['num_episodes_warm']
    train_freq = params['train_freq_warm']
    num_iter_warm = 1

    map_state2index = mapping['state2index']
    map_index2state = mapping['index2state']
    map_index2action = mapping['index2action']
    map_action2index = mapping['action2index']

    for iter in range(num_iter_warm):

        conv = simulator.sequential(reset=True)

        utterance = conv['utterance']
        episode_max = max(num_episodes, len(map_state2index))
        
        for episode in range(episode_max):

            # state_t: user question (plus slots if any: to be implemented)
            s_t = utterance

            if verbose:
                print('%%%%%%%%%%%%%%%%%%%%%%%%')
                print('utterance:', utterance)
            
            if use_Q:
                index_state = map_state2index[s_t]
                index_action, confidence = Q_table.predict(index_state)
                intent = 'Unknown'
                action = map_index2action[index_action]
            else:
                action, intent, confidence = agent.rule_policy(utterance)

            reward = confidence

            index_state = map_state2index[s_t]
            index_action = map_action2index[action]

            if use_Q == False:
                Q_table.add(confidence, index_state, index_action)

            episode_over = 0

            conv = simulator.sequential()

            if conv is not None:

                utterance = conv['utterance'] #can be different from 1 in slots
                s_t_plus1 = utterance

                episode_over = True

                if verbose:
                    print('episode #', episode)
                    print('s_t:', s_t, map_state2index[s_t])
                    print('action:', action, map_action2index[action])
                    print('intent:', intent)
                    print('s_t_plus1:', s_t_plus1)
                    print('reward:', reward)
                    print()

                fill_buffer_all_actions(agent, s_t, action, reward, s_t_plus1, episode_over, warmup=True)

            else:

                break

    # save Q in pickle file
    if use_Q == False:
        Q_table.save_Q()

    print('WARM-UP TRAINING ....')
    agent.clone_dqn = copy.deepcopy(agent.dqn)
    #agent.train(4, 10)
    agent.train(30,50, num_iter = 100)

    return agent, Q_table

def testing(agent, mapping, Q_table, from_Q_table = True, verbose=True, num_test=-1):
    """ Test performance of DQN agent after warm-up training, by comparing responses with NLU responses"""
    
    map_state2index = mapping['state2index']
    map_index2state = mapping['index2state']
    map_index2action = mapping['index2action']
    map_action2index = mapping['action2index']


    if num_test < 0 :
        indices = range(agent.user_act_cardinality)
    else:
        indices = range(num_test)
        

    num_success = 0
    for index in indices:

        example = map_index2state[index]
        rep = np.zeros((1,len(map_index2state)))
        rep[0, index] = 1.0

        index_action_dqn = agent.dqn.predict(rep, {'gamma': agent.gamma})

        if from_Q_table:
            index_watson, _ = Q_table.predict(index)
            action_watson = map_index2action[index_watson]
        else:
            action_watson,_,_ = agent.rule_policy(example)
            index_watson = map_action2index[action_watson]

        if index_watson - index_action_dqn == 0:
            num_success += 1

        if verbose:    
            print('------------------------------------------------------')
            print('input:', example, index)
            print('action NLU:', action_watson, index_watson)
            print('action DQN:', map_index2action[index_action_dqn])
            print('------------------------------------------------------')

    print('In warm-up: # success %d, success rate %f' %( num_success, float(num_success)/float(len(indices))))

    return

def fill_buffer_all_actions(agent, s_t, action, reward, s_t_plus1, episode_over, warmup=False, reward_thre = 0.5, flush=False):
    """ Fill agent replay  buffer, performing data augmentation on more confident cases""" 

    # fill buffer for actual state-action pair
    agent.register_experience_replay_tuple(s_t, action, reward,  s_t_plus1, episode_over, flush=flush)

    do_augmentation = False
    
    if warmup: #always do augmentation in warmup
        do_augmentation = True
    else: #in episodes do augmentation only for high reward (high confidence in action)
        if reward > reward_thre:
            do_augmentation = True
          
    if do_augmentation:
        
        # fill buffer for other actions, which would get low reward
        for act in agent.map_action2index.keys():

            if act != action:
                zero_reward = 0. 
                agent.register_experience_replay_tuple(s_t, act, zero_reward,  s_t_plus1, episode_over, flush=False)

    return


def save_model(path, agent, epoch):
    """ save only model params need to be added """
    filename = 'agt_%d.p' % (epoch)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    checkpoint['model'] = copy.deepcopy(agent.dqn.model)
    #checkpoint['params'] = params
    try:
        pickle.dump(checkpoint, open(filepath, "wb"))
        print('saved model in %s' % (filepath, ))
    except Exception as e:    
        print('Error: Writing model fails: %s' % (filepath, ))
        print(e)

        
