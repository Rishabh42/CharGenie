import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### PROVIDED CODE #####

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}

# modify build_word_counts for SNLI
# so that it takes into account batch['premise'] and batch['hypothesis']
def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in batch:
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####

class CharSeqDataloader():
    def __init__(self, filepath, seq_len, examples_per_epoch):
        with open(filepath, 'r', encoding='utf-8') as file:
            self.data = list(file.read())

        self.unique_chars = list(set(self.data))
        self.vocab_size = len(self.unique_chars)
        self.mappings = self.generate_char_mappings(self.unique_chars)
        self.seq_len = seq_len
        self.examples_per_epoch = examples_per_epoch

        # your code here
    
    def generate_char_mappings(self, uq):
        char_to_idx = {char: idx for idx, char in enumerate(self.unique_chars)}
        idx_to_char = {idx: char for idx, char in enumerate(self.unique_chars)}
        
        return {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}

    def convert_seq_to_indices(self, seq):

        return [self.mappings['char_to_idx'][char] for char in seq if char in self.mappings['char_to_idx']]

    def convert_indices_to_seq(self, seq):
        
        return [self.mappings['idx_to_char'][index] for index in seq if index in self.mappings['idx_to_char']]

    def get_example(self):
        data_indices = self.convert_seq_to_indices(self.data)

        # Calculate the number of characters to loop through for the given number of examples per epoch
        num_characters = len(data_indices) - self.seq_len

        for _ in range(self.examples_per_epoch):
            # Randomly choose a start index for the sequence
            start_index = random.randint(0, num_characters - 1)
            end_index = start_index + self.seq_len + 1

            # Slice the sequence from the dataset
            seq_slice = data_indices[start_index:end_index]
            
            # Prepare the input and target sequences
            in_seq = torch.tensor(seq_slice[:-1], dtype=torch.int64).to(device)
            target_seq = torch.tensor(seq_slice[1:], dtype=torch.int64).to(device)

            yield in_seq, target_seq


class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars

        self.embedding_size = embedding_size

        # your code here
        
    def rnn_cell(self, i, h):
        # your code here
        pass

    def forward(self, input_seq, hidden = None):
        # your code here
        pass

    def get_loss_function(self):
        # your code here
        pass

    def get_optimizer(self, lr):
        # your code here
        pass
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        # your code here
        pass

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        #  your code here

    def forward(self, input_seq, hidden = None, cell = None):
        # your code here
        pass

    def lstm_cell(self, i, h, c):
        # your code here
        pass

    def get_loss_function(self):
        # your code here
        pass

    def get_optimizer(self, lr):
        # your code here
        pass
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        # your code here
        pass

def top_k_filtering(logits, top_k=40):
    # your code here
    pass

def top_p_filtering(logits, top_p=0.9):
    # your code here
    pass

def train(model, dataset, lr, out_seq_len, num_epochs):

    # code to initialize optimizer, loss function

    n = 0
    running_loss = 0
    for epoch in range(num_epochs):
        for in_seq, out_seq in dataset.get_example():
            # main loop code

            n += 1

        # print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss/n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly

        with torch.no_grad():
            pass

        print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    
    return None # return model optionally


def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10 # one epoch is this # of examples
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)

def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)


def fix_padding(batch_premises, batch_hypotheses):
    pass # your code here


def create_embedding_matrix(word_index, emb_dict, emb_dim):
    pass

def evaluate(model, dataloader, index_map):
    pass

class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # your code here

    def forward(self, a, b):
        pass # your code here


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # your code here

    def forward(self, a, b):
        pass # your code here

def run_snli(model):
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    glove_embeddings = "" # fill in your code

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders

    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    # training code

def run_snli_lstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

def run_snli_bilstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

if __name__ == '__main__':
    run_char_rnn()
    # run_char_lstm()
    # run_snli_lstm()
    # run_snli_bilstm()
