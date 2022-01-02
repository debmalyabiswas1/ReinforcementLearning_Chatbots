
from deep_dialog.nlu.watson import  watson
from deep_dialog.nlu.rasa import  rasa

def set_nlu(params, nlu_type, top_intents, **kargs):

        
        print('-----------  NLU in use is %s -----------'%nlu_type)
        
        if nlu_type == 'watson':
            nlu = watson(params, top_intents, **kargs)
        elif nlu_type == 'rasa':
            nlu = rasa(params, top_intents, **kargs)

        return nlu
