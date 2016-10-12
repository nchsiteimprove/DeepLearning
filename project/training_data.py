import json
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

def load_training_data():
    got_data()

    with open(unpacked_data_path) as f:
        raw_training_data = json.loads(f.read())
    training_data = []

    for training_example in raw_training_data:
        if not discard_example(training_example):
            training_data.append(training_example)

    print("Loaded %d training examples"%len(training_data))
    return training_data

def convert_training_data():
    # example_data = {}
    # example_data['html'] = training_example['html']
    # example_data['blocks'] = training_example['blocks']
    pass

def get_attributes_from_data(raw_training_data=None):
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

# with open(unpacked_data_path) as f:
#     dump = json.loads(f.read())
#     print(len(dump))
#     print(dump[0].keys())
#     print(dump[1]['attributes'])
data_filtered = load_training_data()
get_attributes_from_data(data_filtered)
