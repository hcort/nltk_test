import json
import os
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def is_valid(word, stop_words):
    return len(word) > 1 and (word not in stop_words)


def parse_json(json_folder):
    word_cloud = {}
    tokenizer = WordPunctTokenizer()
    stop_words = set(stopwords.words('spanish_2'))
    spanish_stemmer = SnowballStemmer('spanish')
    num_mensajes = 0
    num_palabras = 0
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
                    num_mensajes += 1
                    msg_tokenizado = tokenizer.tokenize(all_messages[message].get('message'))
                    msg_to_lower = [word.lower() for word in msg_tokenizado]
                    msg_filtrado = [word for word in msg_to_lower if is_valid(word, stop_words)]
                    for word in msg_filtrado:
                        num_palabras += 1
                        word_stem = spanish_stemmer.stem(word)
                        stem_dict = word_cloud.get(word_stem, {})
                        if not stem_dict:
                            stem_dict['full_words'] = {}
                        stem_dict['total'] = stem_dict.get('total', 0) + 1
                        stem_dict['full_words'][word] = stem_dict.get('full_words', {}).get(word, 0) + 1
                        word_cloud[word_stem] = stem_dict
    low_pass_filter = []
    for word in word_cloud:
        if word_cloud[word]['total'] < 10:
            low_pass_filter.append(word)
    for word in low_pass_filter:
        word_cloud.pop(word)
    sorted_cloud = sorted(word_cloud.items(), key=lambda item: item[1]['total'], reverse=True)
    sorted_cloud = {k: v for k, v in sorted_cloud}
    print(sorted_cloud)