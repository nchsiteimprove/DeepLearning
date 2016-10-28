import json
import math
import numpy as np
import os
import traceback
import tarfile
import random

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

def slice_list(orig_list, slice_size):
    return [orig_list[i:i + slice_size] for i in xrange(0, len(orig_list), slice_size)]

def load_training_data(n_examples=None, seed=None):
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
            training_example['id'] = str(c)
            training_data.append(training_example)
            c += 1
            if n_examples is not None and c >= n_examples:
                break

    if seed is not None:
        random.seed(seed)
        random.shuffle(training_data)
        print("Shuffled training pages using seed: %s"%str(seed))
    print("Loaded %d training pages"%len(training_data))
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

    total_block_length_orig = 0
    nr_blocks_orig = 0
    total_block_length_chop = 0
    nr_blocks_chop = 0

    global g_chop_blocks
    global g_max_block_length
    global verbose
    global max_encoding

    global content_examples
    global boilerplate_examples
    content_examples = 0
    boilerplate_examples = 0

    code_max = max_encoding - 1

    if encode:
        print("\nEncoding training data...")
    for example in data_filtered:
        html = example['html'].encode(encoding='utf-8')

        block_count = 0

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
                        label = [1.0]#, 0.0]
                    elif b_label == 0:
                        label = [0.0]#, 1.0]

                    total_block_length_orig += len(codes)
                    nr_blocks_orig += 1

                    block_id = example['id'] + '-' + str(block_count)

                    # Some blocks are very long, split them up
                    if g_chop_blocks and len(codes) > g_max_block_length:
                        # print("Slicing blocks")
                        nr_slices = math.ceil(len(codes) / float(g_max_block_length))
                        slice_size = len(codes) / int(nr_slices)

                        if slice_size > g_max_block_length:
                            print("ARRG!!")

                        slices = slice_list(codes, slice_size)

                        slice_count = 0
                        for s in slices:
                            total_block_length_chop += len(s)
                            nr_blocks_chop += 1
                            longest_block = max(longest_block, len(s))
                            shortest_block = min(shortest_block, len(s))

                            slice_id = block_id + '-' + str(slice_count)

                            data_blocks.append({'data': np.array(s).astype('int32'), 'label': np.array(label), 'id': slice_id})
                            slice_count += 1

                            if b_label == 1:
                                content_examples += 1
                            if b_label == 0:
                                boilerplate_examples += 1
                    else: # All blocks retain their size
                        # print("Keeping original block")
                        longest_block = max(longest_block, len(codes))
                        shortest_block = min(shortest_block, len(codes))
                        data_blocks.append({'data': np.array(codes).astype('int32'), 'label': np.array(label), 'id': block_id})
                        if b_label == 1:
                            content_examples += 1
                        if b_label == 0:
                            boilerplate_examples += 1
                    block_count += 1
                else:
                    longest_block = max(longest_block, len(b_html))
                    shortest_block = min(shortest_block, len(b_html))
                    data_blocks.append({'label': b_label, 'data':b_html})


    if encode:
        for data_block in data_blocks:
            data = data_block['data']
            diff_from_longest = longest_block - len(data)
            padding = np.zeros(diff_from_longest, dtype=data.dtype)
            padded_data = np.concatenate([data, padding])
            data_block['data'] = padded_data
            padded_mask = np.concatenate([np.ones(len(data)).astype('int32'), padding.astype('int32')])
            data_block['mask'] = padded_mask

    global g_longest_block
    g_longest_block = longest_block

    if index_less > 0:
        print("WARN: index less than html seen %d times"%index_less)
    if index_beyond > 0:
        print("WARN: index beyond html seen %d times"%index_beyond)
    if statistics:
        print("Longest block length: %d"%longest_block)
        print("Shortest block length: %d"%shortest_block)

    global i_train_end
    i_train_end = len(data_blocks)
    if verbose:
        print("Generated %d training examples"%i_train_end)
        print("Average original block length: %d"%(total_block_length_orig/nr_blocks_orig))
        if g_chop_blocks:
            print("Chopped blocks to max length: %d"%g_max_block_length)
            print("Average chopped block length: %d"%(total_block_length_chop/nr_blocks_chop))
    return data_blocks, encodings

def encoding_statistics(encodings, take=5):
    print("\nEncoded %d tokens"%len(encodings))
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

def expand_data(content_frac = 0.5, seed = None):
    global content_examples
    global boilerplate_examples
    global data_blocks_encoded

    print("\nExpanding data set...")

    content_sample_multpl = int(round((content_frac * boilerplate_examples) / (content_examples * (1 - content_frac))))

    expanded_data = []
    content_examples = 0
    boilerplate_examples = 0

    for d_block in data_blocks_encoded:
        if d_block['label'][0] == 1.0:
            for _ in xrange(content_sample_multpl):
                expanded_data.append(d_block)
                content_examples += 1
        else:
            expanded_data.append(d_block)
            boilerplate_examples += 1

    if seed is not None:
        random.seed(seed)
        random.shuffle(expanded_data)

    data_blocks_encoded = expanded_data

def get_batch(num_blocks, return_ids = False):
    global data_blocks_encoded
    global g_longest_block
    global i_train_end
    global i_train_current
    global verbose

    take = i_train_current + num_blocks

    batch_size = min(num_blocks, i_train_end - i_train_current)

    try:
        X = np.zeros(shape=(batch_size, g_longest_block)).astype('int32')
        y = np.zeros(shape=(batch_size, 1)).astype('int32')
        Xmask = np.zeros(shape=(batch_size, g_longest_block)).astype('int32')
        Ids = []
    except:
        if verbose:
            traceback.print_exc
        return None, None, None

    idx = 0
    while i_train_current < i_train_end and i_train_current < take:
        example = data_blocks_encoded[i_train_current]
        X[idx] = example['data']
        y[idx] = example['label']
        Xmask[idx] = example['mask']
        Ids.append(example['id'])
        i_train_current += 1
        idx += 1

    if return_ids:
        return X, y, Xmask, Ids
    return X, y, Xmask

def reset_batches():
    global i_train_current
    i_train_current = 0

def print_content_ratio():
    global content_examples
    global boilerplate_examples

    print("\nContent examples: %d"%content_examples)
    print("Boilerplate examples: %d"%boilerplate_examples)
    print("%.2f%% content"%((float(content_examples)/(content_examples + boilerplate_examples)) * 100))

verbose = True
max_encoding = 255 + 1
g_longest_block = 0
g_max_block_length = 1000
g_chop_blocks = True
i_train_current = 0
i_train_end = 0
content_examples = None
boilerplate_examples = None

data_filtered = load_training_data(seed=133742)

# print("Page ids:")
# for d in data_filtered:
    # print(d['id'])

data_blocks_encoded, encodings = convert_training_data_individual_blocks(data_filtered, encode=True, statistics=True)

print_content_ratio()

expand_data()

print_content_ratio()

# print("Block ids:")
# for d in data_blocks_encoded:
    # print(d['id'])

# encoding_statistics(encodings)

# sample = data_blocks_encoded[663:666]
# for s in sample:
#     print(s)
