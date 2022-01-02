import watson_developer_cloud

class WatsonAPI(object):

    def __init__(self, WID, username, password):

        self.WID = WID
        self.config_watsonAPI(self.WID, username, password)
        self.ncalls = 0

        return

    def config_watsonAPI(self, WID, username, password):

        #assistant = watson_developer_cloud.AssistantV1(
        #url='https://gateway.watsonplatform.net/assistant/api',
        #username=username,
        #password=password,
        #version='2018-10-00'
        #)

        assistant = watson_developer_cloud.AssistantV1(
        #url='https://gateway-fra.watsonplatform.net/assistant/api/v1/workspaces/34c62bb8-d068-47dd-93c4-609843cad558/message',
        url='https://gateway-fra.watsonplatform.net/assistant/api',
        username=username,
        password=password,
        version='2018-10-00'
        )

        self.assistant = assistant

        return

    def get_intents(self):

        response = self.assistant.list_intents(workspace_id = self.WID).get_result()
        return response

    def get_questions(self, intent):

        response = self.assistant.list_examples(workspace_id = self.WID, intent = intent).get_result()
        return response


    def get_nodes(self, page_limit= 500):

        response = self.assistant.list_dialog_nodes(workspace_id = self.WID, page_limit = page_limit).get_result()

        return response

    def get_node_by_condition(self, intent):

        response = self.assistant.list_dialog_nodes(workspace_id = self.WID).get_result()

        return

    def get_entities(self):

        response = self.assistant.list_entities(workspace_id = self.WID).get_result()

        return response

    def get_entity(self, entity):

        response = self.assistant.get_entity(workspace_id = self.WID,
                entity = entity, export= True).get_result()

        return response

    def message(self, input):

        response = self.assistant.message(
        workspace_id= self.WID, input={'text': input}
        ).get_result()

        self.ncalls += 1

        return response

    def create_intent(self, intent, examples):

        response = self.assistant.create_intent(
        workspace_id = self.WID,
        intent = intent,
        examples = examples
        ).get_result()

        return response

    def delete_intent(self, intent):

        response = self.assistant.delete_intent(
            workspace_id = self.WID,
            intent = intent).get_result()

        return response

    def create_node(self, node):

        kargs = {}
        for key in node.keys():
            if key != 'dialog_node' and key != 'previous_sibling':
                kargs[key] = node[key]


        response = self.assistant.create_dialog_node(
            self.WID, node['dialog_node'], **kargs).get_result()


        return response

    def create_entity(self, entity):

        response = self.assistant.create_entity(self.WID, entity = entity['entity'],
                values = entity['values'])

        return response

    def delete_node(self, node):

        response = self.assistant.delete_dialog_node(
            workspace_id = self.WID,
            dialog_node = node['dialog_node']).get_result()

        return response
