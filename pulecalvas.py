import json
import os
from operator import mod

import hunspell
import requests
import spacy
from bs4 import BeautifulSoup
from nltk import WordPunctTokenizer, SnowballStemmer
from nltk.corpus import stopwords
from syltippy import syllabize
import pickle

from main import json_folder_kaplane

"""
    A simple tool to do some NLP in spanish.
    
    This algorithm find "pulecalvas" in a corpus of text messages extracted from a website.
    
    A word is "pulecalvas" if it's formed of two parts:
        - third person of the present of indicative 
        - a noun
        
    Examples:
        - pulecalvas = pule (pulir) calvas
        - pisacharcos = pisa (pisar) charcos
        - vendemotos = vende (vender) motos
        
    The parameter json folder is the path to our corpus. The expected input are .json files
    with this structure:    
        {
            "parsed_messages": {
                msg_id: {
                    "message": text to parse
                },
                ...
            }
        }
        
    To find these words we will use:
        - nltk for tokening, filtering out stopwords and stemming
        - spacy for POS tagging
        - hunspell for word checking
        
    We also need a list of spanish verbs as nltk si missing a lot of tools for spanish processing.
    
    To build this list of verbs we will use the wiktionary list of spanish verbs.
    We will also use wiktionary to get a list of all the third person of the present of indicative of spanish verbs
    
    This lists are created once and stored in pickle files so they can be reused later.

    Added a fix to syltippy to avoid infinite loops in words with umlaut
        ä, ë, ï, ö can be treated as a, e, i, o.
        ü should be treated as an especial case
        OPEN_PLAIN = set(u'aeoäëö')
        CLOSED_PLAIN = set(u'iuï')
        
    As we are using hunspell you will need the es_ES aff and dic files.
    Also you will need to install the hunspell library:
        > sudo apt-get install libhunspell-dev    

"""


def build_spanish_verb_form():
    """
        lee la tercera persona del presente de indicativo de un verbo conjugado en wiktionario
    """
    #
    verbos = get_verb_list()
    s = requests.Session()
    verbos_tercera = set()
    i = 0
    for verbo in verbos:
        html = s.get(f'https://es.wiktionary.org/wiki/{verbo}#Conjugaci%C3%B3n')
        if html.status_code == 200:
            soup = BeautifulSoup(html.text, 'html.parser')
            forma_verbal = soup.select('table.inflection-table tr:nth-of-type(11) td:nth-of-type(3)')
            for forma in forma_verbal:
                i += 1
                if mod(i, 100) == 0:
                    print(f'{i}/{len(verbos)}')
                verbos_tercera.add(forma.text.strip())
    with open('resources/lista_verbos_tercera_persona.pickle', 'wb') as file:
        pickle.dump(verbos_tercera, file)


def build_spanish_verb_list():
    lista_conjugaciones = [
        'https://es.wiktionary.org/wiki/Categor%C3%ADa:ES:Primera_conjugaci%C3%B3n',
        'https://es.wiktionary.org/wiki/Categor%C3%ADa:ES:Segunda_conjugaci%C3%B3n',
        'https://es.wiktionary.org/wiki/Categor%C3%ADa:ES:Tercera_conjugaci%C3%B3n'
    ]
    s = requests.Session()
    verbos = set()
    for conjugacion in lista_conjugaciones:
        current_url = conjugacion
        while current_url:
            print(f'Cargando {current_url}. Verbos encontrados {len(verbos)}')
            html = s.get(current_url)
            soup = BeautifulSoup(html.text, 'html.parser')
            link_list = soup.select('div#mw-pages > a')
            current_url = ''
            for link in link_list:
                if link.text == 'página siguiente':
                    current_url = 'https://es.wiktionary.org/' + link.attrs.get('href', '')
            entries_list = soup.select('div#mw-pages div.mw-content-ltr  ul  li  a')
            for entry in entries_list:
                if entry.text.endswith('se'):
                    verbos.add(entry.text[:-2].strip())
                else:
                    verbos.add(entry.text.strip())
    with open('resources/lista_verbos_esp.pickle', 'wb') as file:
        pickle.dump(verbos, file)


def get_verb_list():
    pickle_file = 'resources/lista_verbos_esp.pickle'
    if not os.path.exists(pickle_file):
        build_spanish_verb_list()
    with open(pickle_file, 'rb') as file:
        return pickle.load(file)


def get_verb_form_list():
    pickle_file = 'resources/lista_verbos_tercera_persona.pickle'
    if not os.path.exists(pickle_file):
        build_spanish_verb_form()
    with open(pickle_file, 'rb') as file:
        return pickle.load(file)


def is_valid(word, stop_words):
    return len(word) > 1 and (word not in stop_words)


verb_lens = [2, 3, 4]
verb_conj = ['ar', 'er', 'ir']


def build_candidates(syllables, spanish_stemmer, spanish_verbs, spanish_verbs_form):
    candidates = []
    for verb_len in verb_lens:
        verb = ''.join(syllables[:verb_len])
        verb_root = spanish_stemmer.stem(verb)
        for conj in verb_conj:
            conjugado = f'{verb_root}{conj}'
            if (verb in spanish_verbs_form) and (conjugado in spanish_verbs) and (verb_len < len(syllables) - 1):
                candidates.append({
                    'len': verb_len,
                    'root': verb_root,
                    'conj': conjugado
                })
    return candidates


def find_pulecalvas(json_folder):
    """
        Busco palabras del tipo pulecalvas, abrazafarolas, cierrabares
    """
    tokenizer = WordPunctTokenizer()
    stop_words = set(stopwords.words('spanish_2'))
    spanish_stemmer = SnowballStemmer('spanish')
    spanish_verbs = get_verb_list()
    spanish_verbs_form = get_verb_form_list()
    hunspell_dict = hunspell.HunSpell('resources/es_ES.dic', 'resources/es_ES.aff')
    nlp = spacy.load("es_core_news_sm")
    word_count = {}
    for root, subdirs, files in os.walk(json_folder):
        num_files = len(files)
        for idx, json_id in enumerate(files):
            if json_id.endswith('pickle'):
                continue
            print(f'{idx}/{num_files} - {json_id}')
            json_file = os.path.join(json_folder, json_id)
            with open(json_file, 'r') as file:
                thread_dict = json.load(file)
                all_messages = thread_dict.get('parsed_messages', {})
                for message in all_messages:
                    msg_tokenizado = tokenizer.tokenize(all_messages[message].get('message'))
                    msg_to_lower = [word.lower() for word in msg_tokenizado]
                    pos_sentence = nlp(' '.join(msg_tokenizado))
                    msg_filtrado = [(idx, word) for idx, word in enumerate(msg_to_lower) if is_valid(word, stop_words)]
                    for idx, word in msg_filtrado:
                        # fixme
                        if (not hunspell_dict.spell(word)) and (len(word) > 7):
                            syllables, stress = syllabize(word)
                            candidates = build_candidates(syllables, spanish_stemmer, spanish_verbs, spanish_verbs_form)
                            for candidate in candidates:
                                verbo = ''.join(syllables[:candidate['len']])
                                sustantivo = ''.join(syllables[candidate['len']:])
                                doc = nlp(sustantivo)
                                pos_simple_str = doc[0].pos_
                                pos_in_sentence_str = pos_sentence[idx].pos_
                                if hunspell_dict.spell(sustantivo):
                                    print(f'{word} = {verbo}-{sustantivo} - '
                                          f'{candidate["root"]}/{candidate["conj"]} - {sustantivo} = {pos_simple_str} - {word} = {pos_in_sentence_str}')
                                    word_count[word] = word_count.get(word, 0) + 1
    print(word_count)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # build_spanish_verb_list()
    build_spanish_verb_form()
    # find_pulecalvas()
