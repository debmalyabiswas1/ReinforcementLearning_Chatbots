from deep_dialog.nlu.watson_handler import WatsonAPI

class watson():

    def __init__(self, params, top_intents):

        self.params = params
        self.top_intents = top_intents
        
        username= 'apikey'
        password = 'J7gDnryfiw_ucA7FzqlSiwI3lpnFuIHvq2VG6Ue0y9SZ'
        WID ='5ad83f0a-2e75-4a7d-934b-2db3aad2f5db' #'34c62bb8-d068-47dd-93c4-609843cad558'
        self.watson = WatsonAPI(WID, username, password)

    def predict(self, utterance):

        response = self.watson.message(utterance)
        action = response['output']['text'][0]
        try:
            intent = response['intents'][0]['intent']
            confidence = response['intents'][0]['confidence']
        except:
            intent = None
            confidence = 0.
            print('Error in reading intent from response', response )

        return action, intent, confidence

    def actions_to_dict(self):

        response = self.watson.get_nodes()

        count = 0
        map_index2action = {}
        map_action2index = {}

        for node in response['dialog_nodes']:

            if 'generic' in node['output']: #Anything_else node

                output_list = node['output']['generic'][0]['values']
                n_act_fallback = len(output_list)

                for iact, item in enumerate(output_list):

                    action = item['text']

                    map_action2index[action] = count
                    map_index2action[count] = action
                    count += 1


            elif 'text' not in node['output']: #Feedback node

                continue

            else:

                n_act =  len(node['output']['text']['values'])

                for iact, action in enumerate(node['output']['text']['values']):

                    action = action.replace('\\','')
                    map_action2index[action] = count
                    map_index2action[count] = action
                    count += 1

        return map_index2action, map_action2index
