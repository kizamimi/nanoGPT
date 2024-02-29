"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests, zipfile, io
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://mattmahoney.net/dc/enwik8.zip'
    r = requests.get(data_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.path.dirname(__file__))
    unzip_file_path = os.path.join(os.path.dirname(__file__), 'enwik8')
    os.rename(unzip_file_path, input_file_path)

with open(input_file_path, 'r', encoding="utf-8") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

stop_num = 100
new_data = ""
for i in range(len(data)):
    if ord(data[i]) >= ord("a") and ord(data[i]) <= ord("z"):
        new_data += data[i]
    elif ord(data[i]) >= ord("A") and ord(data[i]) <= ord("Z"):
        new_data += data[i].lower()
    elif data[i] == " ":
        new_data += data[i]
    elif data[i] == "0":
        new_data += "zero "
    elif data[i] == "1":
        new_data += "one "
    elif data[i] == "2":
        new_data += "two "
    elif data[i] == "3":
        new_data += "three "
    elif data[i] == "4":
        new_data += "four "
    elif data[i] == "5":
        new_data += "five "
    elif data[i] == "6":
        new_data += "six "
    elif data[i] == "7":
        new_data += "seven "
    elif data[i] == "8":
        new_data += "eight "
    elif data[i] == "9":
        new_data += "nine "
    else:
        pass
    # if i % 100 == 0 and i != 0:
    #     print("\r"+new_data,end="")
    #     input()

data = new_data

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):int(n*0.95)]
test_data = data[int(n*0.95):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  99,621,824
# all the unique characters:
# " abcdefghijklmnopqrstuvwxyz"
# vocab size: 27
# train has 85,959,602 tokens
# val has 4,775,533 tokens
# test has 4,775,534 tokens
