import pandas as pd
import tweetnlp
import sys
model = tweetnlp.load_model('ner')  # Or `model = tweetnlp.NER()`
model.ner('Jacob Collier is a Grammy-awarded English artist from London.', return_probability=True)  # Or `model.predict`
import spacy
nlp = spacy.load('en_core_web_lg')
import re
filename = sys.argv[1]
df=pd.read_csv(f'chunked_files/{filename}')
def resolve_names(text):
    doc = nlp(text)
    name_dict = {}
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            name = ent.text.strip()
            if ' ' in name:
                first_name, last_name = name.split(' ', 1)
                if first_name in name_dict:
                    if len(name) > len(name_dict[first_name]):
                        name_dict[first_name] = name
                else:
                    name_dict[first_name] = name
            else:
                for full_name in name_dict.values():
                    if name in full_name:
                        name_dict[name] = full_name
                        break
                else:
                    name_dict[name] = name
    for i, full_name in enumerate(name_dict.values()):
        name_dict[list(name_dict.keys())[i]] = re.sub(r'[^a-zA-Z\s].*', '', full_name)
    unique_names = list(set(name_dict.values()))
    final_names = []
    for name in unique_names:
        is_substring = False
        for other_name in unique_names:
            if name != other_name and name in other_name:
                is_substring = True
                break
        if not is_substring:
            final_names.append(name)
    return final_names
def get_names(strx):
  x=model.ner(strx, return_probability=True)
  namelist=[]
  for i in x:
    if i["type"]=="person" and i["probability"]>0.5:
      namelist.append(i["entity"])
  return namelist
spacy_lis=[]
twit_lis=[]
from tqdm import tqdm
for i in tqdm(df["translated ad_creative_bodies"]):
    if pd.isna(i):
        spacy_lis.append([])
        twit_lis.append([])
    else:
        x = resolve_names(i)
        spacy_lis.append(x)
        y = get_names(i)
        twit_lis.append(y)
df["spacy_ner"]=spacy_lis
df["twit_ner"]=twit_lis
df.to_csv(f"named_files/names_{filename}")