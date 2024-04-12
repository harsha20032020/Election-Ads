import sys
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer.utils import preprocess_batch, postprocess_batch
from IndicTransTokenizer.tokenizer import IndicTransTokenizer
#if len(sys.argv) > 1:
#    quantization = sys.argv[1]
#else:
quantization = ""
#en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B" # ai4bharat/indictrans2-en-indic-dist-200M
#indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B" # ai4bharat/indictrans2-indic-en-dist-200M
indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-dist-200M"
#indic_indic_ckpt_dir = "ai4bharat/indictrans2-indic-indic-1B"   # ai4bharat/indictrans2-indic-indic-dist-320M
BATCH_SIZE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
        revision="8b13e3499f07004b60f99698377e751e0dd4516e",
    )
    
    if qconfig==None:
        model = model.to(DEVICE)
    #    model.half()
    
    model.eval()
    
    return tokenizer, model

from tqdm import tqdm
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer):
    translations = []
    for i in tqdm(range(0, len(input_sentences), BATCH_SIZE)):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch, entity_map = preprocess_batch(
            batch, src_lang=src_lang, tgt_lang=tgt_lang
        )

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(), src=False
        )

        # Postprocess the translations, including entity replacement
        translations += postprocess_batch(
            generated_tokens, lang=tgt_lang, placeholder_entity_map=entity_map
        )

        del inputs
        torch.cuda.empty_cache()

    return translations
indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(
    indic_en_ckpt_dir, "indic-en", "None"
)
print("Model Loaded")
from collections import Counter
def most_common_indic_language_using_characters(input_string):
    # Define character sets for various Indic scripts
    # Note: These are only subsets of common characters for demonstration purposes
    script_characters = {
        'hin_Deva': set("अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंःँ"), 
        'pan_Guru': set("ਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸਹਾਿੀੁੂੇੈੋੌ੍ਂਃਁ"), 
        'tel_Telu': set("అఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలవశషసహాిీుూృెేైొోౌ్ఁంః"),
        'ben_Beng': set("অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূেৈোৌ্"),
        'tam_Taml': set("அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவளழறன்"),
        'ory_Orya': set("ଅଆଇଈଉଊଋଌଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହଁଂ"),
        'kan_Knda': set("ಅಆಇಈಉಊಋಌಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲವಶಷಸಹಾಿೀುೂೃೆೇೈೊೋೌ್ಂಃ"),
        'mal_Mlym': set("അആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരറലവശഷസഹാിീുൂൃെേൈൊോൌ്ംഃ"),
        'guj_Gujr': set("અઆઇઈઉઊઋઍએઐઑઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલવશષસહાિીુૂેૈોૌ્ંઃ"),
    }

    # Function to determine the script of a character
    def get_script(char):
        for script, characters in script_characters.items():
            if char in characters:
                return script
        return None

    # Counting the scripts
    script_counter = Counter(get_script(char) for char in input_string if get_script(char) is not None)

    if not script_counter:
        return "English"

    # Finding the most common script
    most_common_script = script_counter.most_common(1)[0][0]

    return most_common_script
import pandas as pd
import time
start = time.time()
import pandas as pd
file_processing = "chunk_91-95"
df = pd.read_csv(f'chunked_files/{file_processing}.csv')
end = time.time()
print("Read csv with chunks: ",(end-start),"sec")
print(file_processing)
from tqdm import tqdm
def batch_translate_dataframe_largecol(df,column):
    grouped_sentences = {}
    for i, name in enumerate(df[column]):
        src_lang = most_common_indic_language_using_characters(str(name))
        if src_lang=="hin_Deva":
            if df["languages"][i]=="hi":
                src_lang="hin_Deva"
            if df["languages"][i]=="mr":
                src_lang="mar_Deva"
        if src_lang=="ben_Beng":
            if df["languages"][i]=="bn":
                src_lang="ben_Beng"
            if df["languages"][i]=="as":
                src_lang="asm_Beng"
        if src_lang == "English":
            # If the language is already English, no translation needed
            if "English" not in grouped_sentences:
                grouped_sentences["English"] = []
            grouped_sentences["English"].append((i, name))
        else:
            if src_lang not in grouped_sentences:
                grouped_sentences[src_lang] = []
            grouped_sentences[src_lang].append((i, name))
            
    for src_lang in grouped_sentences.keys():
        for i in range(len(grouped_sentences[src_lang])):
            idx, sentence = grouped_sentences[src_lang][i]
            if pd.isna(sentence):
                grouped_sentences[src_lang][i] = (idx, "")
            else:
                grouped_sentences[src_lang][i] = (idx, sentence[:2048])
            
    translations = [None] * len(df)
    perceived_languages = [None] * len(df)
    for src_lang, sentences in grouped_sentences.items():
        indices, sentences_to_translate = zip(*sentences)
        for idx in indices:
            perceived_languages[idx] = src_lang
        if src_lang != "English":
            translated_sentences = batch_translate(sentences_to_translate, src_lang, "eng_Latn",indic_en_model, indic_en_tokenizer)
        else:
            translated_sentences = sentences_to_translate

        for idx, translation in zip(indices, translated_sentences):
            translations[idx] = translation
    df["translated "+column]=translations
    if column=="ad_creative_bodies":
        df["percieved_translation"]=perceived_languages
    return df
df=batch_translate_dataframe_largecol(df,"page_name")
print("Page Names Done")
df=batch_translate_dataframe_largecol(df,"ad_creative_bodies")
df.to_csv(f"translated_chunked_files/translated_{file_processing}.csv")
