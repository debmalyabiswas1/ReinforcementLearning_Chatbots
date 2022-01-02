from rasa_nlu.model import Interpreter
from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
import json
import time
import os
import shutil


class rasa():
    """ 
    Class of template NLU which should contain at least the train and predict methods
    Other methods like add_utterance have being added to test NLU policy updates (not in use in the last version of the code)
    -------------------------------------
    Methods:
    train(): train NLU with training set given in congif file
    predict(utterance): predict NLU response
         Return: message, intent, confidence
    """
    
    def __init__(self, params, top_intents, config_file="./deep_dialog/nlu//rasa_config/tensorflow_config.yml", training_file='./deep_dialog/nlu/rasa_config/GStraining.json'):
        """ Load rasa config from config files and train NLU """

        # do first training
        self.params = params
        self.top_intents = top_intents
        self.config_file = config_file

        self.no_intent = 'No intent detected'
        self.cl_threshold = 0.1

        # initial training data
        with open(training_file, 'r') as f:
            self.training_dict = json.load(f)

        training_data_init = load_data(training_file)
        self.training_dir = os.path.dirname(training_file)

        self.training_data = training_data_init
        self.temp_training_file = self.training_dir + '/GStraining_tmp.json'
        self.training_file = training_file
        if training_file != self.temp_training_file:
            shutil.copy(training_file, self.temp_training_file)

        # training
        self.train()
        print('Training in rasa done !')

        # init rasa interpreter, which is used for prediction
        self.interpreter = Interpreter.load(self.model_directory)

    def train(self):
        """ Train NLU """
        
        
        trainer = Trainer(config.load(self.config_file))
        trainer.train(self.training_data)
        self.model_directory = trainer.persist('./projects/default/')

        return

    def predict(self, utterance):
        """ predict action (with confidence) given a state (utterance) """
        
        fallback = 'No intent detected'
        result = self.interpreter.parse(utterance)
        intent = result['intent']['name']
        confidence = result['intent']["confidence"]

        if confidence < self.cl_threshold:
            intent = self.no_intent
            confidence = 1.-confidence

        message = intent

        return message, intent, confidence 

    def add_utterance(self, utterance, intent, save_json = True):

        examples = self.training_dict['rasa_nlu_data']['common_examples']

        examples.append(self.make_ex_dict(utterance, intent))

        self.training_dict['common_examples'] = examples

        training_dict = {}
        training_dict['common_examples'] = examples
        training_dict['entity_synonyms'] = self.training_dict['rasa_nlu_data']['entity_synonyms']

        data = {
            "rasa_nlu_data": training_dict
        }

        if save_json:
            with open(self.temp_training_file, 'w') as fp:
                json.dump(data, fp, indent=2)
                #print('Training set written to %s' % self.temp_training_file)

    def make_ex_dict(self,utterance, intent):

        ex_dict = {}
        ex_dict['text'] = utterance
        ex_dict['intent'] = intent
        ex_dict["entities"] = []

        return ex_dict

    def actions_to_dict(self):

        print('Creating action to index mapping')

        count = 0
        map_index2action = {}
        map_action2index = {}

        data = self.training_dict
        example_list = data["rasa_nlu_data"]["common_examples"]

        # I take only intents in intent_list
        #if self.top_intents:
        #    example_list = list(set(example_list) & set(self.top_intents))
        intent_list = [item['intent'] for item in example_list]
        u_intent_list = list(set(intent_list))

        if self.no_intent not in u_intent_list:
            u_intent_list.append(self.no_intent)

        count = 0

        for iact, action in enumerate(u_intent_list):
            #in rasa action == intent

            map_action2index[action] = count
            map_index2action[count] = action
            count += 1

        return map_index2action, map_action2index
