import json
import numpy as np
import os
import traceback
import tarfile

unpacked_data_path = 'mldata/TrainingDump.json'

attr_blacklist = ['delete', 'discuss', 'encoding', 'malformed', 'blockifier', 'chardet', 'blockifer']

def got_data():
    if not os.path.exists(unpacked_data_path):
        print("No training data found")
        exit(1)

def discard_example(training_example):
    if training_example['eval_state'] != 2 or training_example['attributes'] is None:
        return True

    bad = [1 for attr in training_example['attributes'] if attr['name'].strip() in attr_blacklist]
    return len(bad) > 0

def load_training_data(n_examples=None):
    got_data()

    with open(unpacked_data_path) as f:
        data = f.read()
        # print(type(data))
        # data = data.encode(encoding='utf-8')
        # print(type(data))
        raw_training_data = json.loads(data, encoding='utf-8')
    training_data = []

    c = 0
    for training_example in raw_training_data:
        if not discard_example(training_example):
            training_data.append(training_example)
            c += 1
            if n_examples is not None and c >= n_examples:
                break

    print("Loaded %d training examples"%len(training_data))
    return training_data

def print_attributes_from_data(raw_training_data=None):
    if raw_training_data is None:
        got_data()

        with open(unpacked_data_path) as f:
            raw_training_data = json.loads(f.read())

    all_attributes = set([])

    total_pages = len(raw_training_data)
    page_count = 0

    for example in raw_training_data:
        page_count += 1
        if example['eval_state'] != 2 or example['attributes'] is None:
            continue
        attributes = set([a['name'] for a in example['attributes']])
        all_attributes |= attributes

    print(", ".join(sorted(all_attributes)))

def print_raw_data_keys(raw_data):
    sample = raw_data[0]
    for key in sample.keys():
        print(key)
        t = type(sample[key])
        print("\t%s"%str(t))
        if t is list:
            inner_sample = sample[key][0]
            inner_t = type(inner_sample)
            print("\t\t%s"%inner_t)
            if inner_t is dict:
                print("\t\t%s"%str(inner_sample.keys()))

def convert_training_data_individual_blocks(data_filtered, encode=True, statistics=False):
    data_blocks = []
    encodings = {}
    longest_block = 0
    shortest_block = 999999999
    index_beyond = 0
    index_less = 0

    code_max = 255

    if encode:
        print("Encoding training data...")
    for example in data_filtered:
        html = example['html'].encode(encoding='utf-8')

        if example['blocks'][-1]['end'] < len(html):
            index_less += 1
        for block in example['blocks']:
            if block['end'] > len(html):
                index_beyond += 1
            b_html = html[block['start']:block['end']].decode(encoding='utf-8')
            if block['ctype'] == 'content':
                b_label = 1
            elif block['ctype'] in ['boilerplate']:
                b_label = 0
            elif block['ctype'] in ['template', 'evaluate']:
                pass
            else:
                b_label = None
                print("Skipping example, no label for type: %s"%block['ctype'])

            if b_label is not None and len(b_html) > 0:
                if encode:
                    codes = []
                    for char in b_html:
                        code = ord(char)

                        if code_max is not None and code > code_max:
                            code = code_max + 1

                        codes.append(code)
                        if code not in encodings:
                            encodings[code] = 1
                        else:
                            encodings[code] += 1
                    if b_label == 1:
                        label = [1.0, 0.0]
                    elif b_label == 0:
                        label = [0.0, 1.0]

                    data_blocks.append({'data': np.array(codes), 'label': np.array(label)})
                else:
                    data_blocks.append({'label': b_label, 'data':b_html})

                if statistics:
                    longest_block = max(longest_block, len(b_html))
                    shortest_block = min(shortest_block, len(b_html))

    if encode:
        default_mask = np.zeros(longest_block).astype('float32')
        for data_block in data_blocks:
            data_block['mask'] = default_mask[:len(data_block['data'])] = np.ones(len(data_block['data'])).astype('float32')

    if index_less > 0:
        print("WARN: index less than html seen %d times"%index_less)
    if index_beyond > 0:
        print("WARN: index beyond html seen %d times"%index_beyond)
    if statistics:
        print("Longest block length: %d"%longest_block)
        print("Shortest block length: %d"%shortest_block)

    return data_blocks, encodings

def encoding_statistics(encodings, take=5):
    print("Encoded %d tokens"%len(encodings))
    sort = sorted(encodings, key=encodings.__getitem__)

    common_tokens = sort[-take:]
    print("Most common tokens:")
    for token in common_tokens[::-1]:
        print("%s: %d"%(unichr(token), encodings[token]))

    least_common_tokens = sort[:take]
    print("Least common tokens:")
    for token in least_common_tokens:
        print("%s: %d"%(unichr(token), encodings[token]))

    ascii_max = 255
    max_c = 0
    max_t = ''
    seenZero = False
    for token, count in encodings.items():
        if token == 0:
            seenZero = True

        if token > ascii_max and count > max_c:
            max_c = count
            max_t = token

    print("Most common non-ascii character: %s, count: %d"%(unichr(max_t), max_c))
    if seenZero:
        print("Saw character with value 0")

def get_batch(n):


data_filtered = load_training_data()
data_blocks_encoded, encodings = convert_training_data_individual_blocks(data_filtered, encode=True, statistics=True)
encoding_statistics(encodings)