
import hashlib
import requests
import pandas as pd
import re
import nltk
#nltk.download('punkt')

# this function can extract NER entites using BERN2
# - given any biomedical or PubMed Abstract,title,keyowrd etc or any text

def clean_text(text):
  """Remove section titles and figure descriptions from text"""
  clean = "\n".join([row for row in text.split("\n") if (len(row.split(" "))) > 3 and not (row.startswith("(a)"))
                    and not row.startswith("Figure")])
  return clean


def query_plain(text, url="http://bern2.korea.ac.kr/plain"):
    return requests.post(url, json={'text': str(text)}).json()

""" function takes one dataframe and one column as input and returns a dictionary by mapping each index with column value"""

def get_dict(df, column):
    dict = {}
    for i in range(len(df)):
        dict[i] = df[column][i]
    return dict


def getentities_fromabstract(abstract):
    entity_list = []
    # The last sentence could be invalid
    text = abstract
    ctext = clean_text(text)
    sentences = nltk.tokenize.sent_tokenize(ctext)

    for s in sentences:
        entity_list.append(query_plain(s))

    parsed_entities = []
    for entities in entity_list:
        e = []
        # If there are not entities in the text
        if not entities.get('annotations'):
            parsed_entities.append({'text':entities['text'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})
            continue
        for entity in entities['annotations']:
            other_ids = [id for id in entity['id'] if not id.startswith("BERN")]
            entity_type = entity['obj']
            entity_name = entities['text'][entity['span']['begin']:entity['span']['end']]
            try:
                entity_id = [id for id in entity['id'] if id.startswith("BERN")][0]
            except IndexError:
                entity_id = entity_name
            e.append({'entity_id': entity_id, 'other_ids': other_ids, 'entity_type': entity_type, 'entity': entity_name})
        parsed_entities.append({'entities':e, 'text':entities['text'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})


    tuple_list_main = []
    tuple_list_sub = []
    try: 
        for i in range(len(parsed_entities)):
            entity = parsed_entities[i]['entities']
            text = parsed_entities[i]['text']
            text_sha = parsed_entities[i]['text_sha256']
            tuple_list_main.append((text, text_sha))
            try:
                for j in range(len(entity)):
                    entity_id_custom = j
                    other_ids = entity[j]['other_ids']
                    entity_type = entity[j]['entity_type']
                    entity_name = entity[j]['entity']
                    tuple_list_sub.append((entity_id_custom, other_ids, entity_type, entity_name))
            except:
                entity_id = 'None found'
                other_ids = 'None found'
                entity_type = 'None found'
                entity_name = 'None found'
            
                
    except:
        pass
    df_text_sha = pd.DataFrame(tuple_list_main, columns=['text', 'text_sha256'])
    df_sub_entity = pd.DataFrame(tuple_list_sub, columns=['entity_id', 'other_ids', 'entity_type', 'entity'])
    dict_text = get_dict(df_text_sha, 'text')
    dict_hash = get_dict(df_text_sha, 'text_sha256')
    df_sub_entity['text'] = df_sub_entity['entity_id'].map(dict_text)
    df_sub_entity['text_sha256'] = df_sub_entity['entity_id'].map(dict_hash)
    return df_sub_entity
    
    
    
    
    
  