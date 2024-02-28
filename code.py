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

        # Define the embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=n_chars, embedding_dim=embedding_size)
        
        # Define the linear layers for the RNN cell
        # W_ax
        self.wax = nn.Linear(in_features=embedding_size, out_features=hidden_size, bias=False)
        # W_aa
        self.waa = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        # W_ya
        self.wya = nn.Linear(in_features=hidden_size, out_features=n_chars, bias=False)
        
        # Define the bias terms for the hidden state and the output separately
        # b_a
        self.ba = nn.Parameter(torch.zeros(hidden_size))
        # b_y
        self.by = nn.Parameter(torch.zeros(n_chars))
        
    def rnn_cell(self, i, h):
        h_next = torch.tanh(self.wax(i) + self.waa(h))

        # Compute the output
        o = self.wya(h_next)

        return o, h_next

    def forward(self, input_seq, hidden = None):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(input_seq.device)
        
        # Embed the whole sequence at once
        embedded = self.embedding_layer(input_seq)
        
        # List to store the outputs at each time step
        outputs = []

        # Process each time step in the input sequence
        for i in range(input_seq.size(0)):
            # Apply rnn_cell to the current input and the previous hidden state
            out, hidden = self.rnn_cell(embedded[i], hidden)
            outputs.append(out)
        
        # Stack the outputs into a single tensor
        out = torch.stack(outputs, dim=0)
        
        # Return the final output sequence and the last hidden state
        return out, hidden

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        import torch.optim as optim
        return optim.Adam(self.parameters(), lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        gen_seq = [starting_char]
        h = torch.zeros(1, self.hidden_size)

        for _ in range(seq_len):
            i = torch.tensor([[gen_seq[-1]]], dtype=torch.long)
            
            output, h = self.forward(i, h)

            # Apply temperature scaling on the logits
            o = output[-1] / temp

            # Apply softmax to get probabilities
            probabs = F.softmax(o, dim=1).squeeze()

            # Sampling from the distribution
            samples = Categorical(probabs).sample().item()
            
            gen_seq.append(samples)

        return gen_seq

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        self.embedding_layer = nn.Embedding(num_embeddings=n_chars, embedding_dim=embedding_size)
        
        self.forget_gate = nn.Linear(in_features=hidden_size + embedding_size, out_features=hidden_size)
        self.cell_gate = nn.Linear(in_features=hidden_size + embedding_size, out_features=hidden_size)

        self.input_gate = nn.Linear(in_features=hidden_size + embedding_size, out_features=hidden_size)
        self.output_gate = nn.Linear(in_features=hidden_size + embedding_size, out_features=hidden_size)
        
        self.cell_state_layer = nn.Linear(in_features=hidden_size + embedding_size, out_features=hidden_size)
        
        self.fc_output = nn.Linear(in_features=hidden_size, out_features=n_chars)

    def forward(self, input_seq, hidden = None, cell = None):
        # your code here
        if hidden is None or cell is None:
            hidden = torch.zeros(self.hidden_size, device=input_seq.device)
            cell = torch.zeros(self.hidden_size, device=input_seq.device)

        outputs = []

        for i in range(input_seq.size(0)):
            i_embedded = self.embedding_layer(input_seq[i])
            output, hidden, cell = self.lstm_cell(i_embedded.squeeze(), hidden, cell)
            outputs.append(output)

        out_seq = torch.stack(outputs).squeeze(1)

        return out_seq, hidden, cell

    def lstm_cell(self, i, h, c):
        input = torch.cat((i, h), dim=0)

        forget_g = torch.sigmoid(self.forget_gate(input))
        
        input_g = torch.sigmoid(self.input_gate(input))

        ct_layer = torch.tanh(self.cell_state_layer(input))

        c_new = forget_g * c + input_g * ct_layer

        out_g = torch.sigmoid(self.output_gate(input))
        h_new = out_g * torch.tanh(c_new)
        o = self.fc_output(h_new)

        return o, h_new, c_new

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        import torch.optim as optim
        return optim.Adam(self.parameters(), lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        starting_index = starting_char

        outputs = [starting_index]
        input_seq = torch.full((1, 1), starting_index, device=device)
        hidden = None
        cell = None

        for index in range(seq_len):
            output, hidden, cell = self.forward(input_seq, hidden, cell)

            logits = output.squeeze(0) / temp

            if top_k is not None:
                logits = top_k_filtering(logits, top_k)
            elif top_p is not None:
                logits = top_p_filtering(logits, top_p)

            proba = F.softmax(logits, dim=0)

            next_char = Categorical(proba).sample().item()

            outputs.append(next_char)
            input_seq = torch.full((1, 1), next_char, device=device)

        return outputs

def top_k_filtering(logits, top_k=40):
    if top_k > 0:
        # For each batch entry, leave the top k logits as they are and set the rest to negative infinity
        top_k_values, _ = torch.topk(logits, k=top_k, dim=-1)
        kth_best = top_k_values[:, -1].view(-1, 1)
        mask = logits < kth_best
        logits[mask] = float('-inf')
    return logits

def top_p_filtering(logits, top_p=0.9):
    # your code here
    probabilities = torch.softmax(logits, dim=-1)
    
    # Sort the probabilities to identify the top p
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    
    # Calculate the cumulative sum of the sorted probabilities
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    
    remove_sorted_indices = cumulative_probabilities > top_p
    # Shift the mask to the right to keep the first token above the threshold
    remove_sorted_indices[..., 1:] = remove_sorted_indices[..., :-1].clone()
    remove_sorted_indices[..., 0] = 0
    
    # Convert the sorted indices back to the original indices
    remove_original_indices = remove_sorted_indices.scatter(dim=-1, index=sorted_indices, src=remove_sorted_indices)
    
    # Apply the mask to the logits, setting tokens to remove to negative infinity
    logits[remove_original_indices] = float('-inf')
    
    return logits

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
    from torch.nn.utils.rnn import pad_sequence

    batch_premises_padded = pad_sequence([torch.tensor(p) for p in batch_premises], batch_first=True)
    batch_hypotheses_padded = pad_sequence([torch.tensor(h) for h in batch_hypotheses], batch_first=True)
    
    batch_premises_reversed = pad_sequence([torch.tensor(p[::-1]) for p in batch_premises], batch_first=True)
    batch_hypotheses_reversed = pad_sequence([torch.tensor(h[::-1]) for h in batch_hypotheses], batch_first=True)
    
    return batch_premises_padded, batch_hypotheses_padded, batch_premises_reversed, batch_hypotheses_reversed


def create_embedding_matrix(word_index, emb_dict, emb_dim):
    import numpy as np

    max_index = max(word_index.values())
    embedding_matrix = np.zeros((max_index + 1, emb_dim))

    for word, i in word_index.items():
        embedding_vector = emb_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

    return torch.from_numpy(embedding_matrix).float()

def evaluate(model, dataloader, index_map):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in dataloader:
            premise = batch['premise']
            hypotheses = batch['hypothesis']
            labels = batch['label']

            premises_indices = []
            hypothesis_indices = []

            for words in premise:
                indices = tokens_to_ix(tokenize(words), index_map)
                premises_indices.append(indices)

            for words in hypotheses:
                indices = tokens_to_ix(tokenize(words), index_map)
                hypothesis_indices.append(indices)

            premises_indices = [[inner[0] for inner in outer if inner] for outer in premises_indices]
            hypothesis_indices = [[inner[0] for inner in outer if inner] for outer in hypothesis_indices]

            pred_output = model.forward(premises_indices, hypothesis_indices)

            true_labels.extend(labels)
            predicted_labels.extend(pred_output.argmax(dim=1).tolist())

    num_correct = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)

    return num_correct / len(true_labels)

class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, device=device)

        self.int_layer = nn.Linear(2*hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

    def forward(self, a, b):
        batch_premises, batch_hypotheses, batch_premises_reversed, batch_hypotheses_reversed = fix_padding(a, b)
        premise_embedding = self.embedding_layer(batch_premises)
        hypothesis_embedding = self.embedding_layer(batch_hypotheses)

        output_premise, (hidden_state_premise, cell_state_premise) = self.lstm(premise_embedding)
        output_hypothesis, (hidden_state_hypothesis, cell_state_hypothesis) = self.lstm(hypothesis_embedding)

        combined = torch.cat((cell_state_premise[-1], cell_state_hypothesis[-1]), dim=1)
        output = self.out_layer(torch.relu(self.int_layer(combined)))

        return output


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.lstm_forward = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bias=True,
                                    batch_first=True, device=device)
        self.lstm_backward = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bias=True,
                                     batch_first=True, device=device)
        self.int_layer = nn.Linear(4 * hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

    def forward(self, a, b):
        batch_premises, batch_hypotheses, batch_premises_reversed, batch_hypotheses_reversed = fix_padding(a, b)
        premise_embedding = self.embedding_layer(batch_premises)
        hypothesis_embedding = self.embedding_layer(batch_hypotheses)
        premise_reversed_embedding = self.embedding_layer(batch_premises_reversed)
        hypothesis_reversed_embedding = self.embedding_layer(batch_hypotheses_reversed)

        output_premise, (hidden_state_premise, cell_state_premise) = self.lstm_forward(premise_embedding)
        output_hypothesis, (hidden_state_hypothesis, cell_state_hypothesis) = self.lstm_forward(hypothesis_embedding)

        output_premise_reversed, (hidden_state_premise_reversed, cell_state_premise_reversed) = self.lstm_backward(
            premise_reversed_embedding)

        output_hypothesis_reversed, (
            hidden_state_hypothesis_reversed, cell_state_hypothesis_reversed) = self.lstm_backward(
            hypothesis_reversed_embedding)

        combined = torch.cat(
            (cell_state_premise[-1], cell_state_premise_reversed[-1], cell_state_hypothesis[-1],
             cell_state_hypothesis_reversed[-1]),
            dim=1)
        output = self.out_layer(torch.relu(self.int_layer(combined)))

        return output

def run_snli(model):
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    glove_embeddings = glove.to_numpy()

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders
    dataloader_train = DataLoader(train_filtered, batch_size=32, shuffle=True)
    dataloader_valid = DataLoader(valid_filtered, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(test_filtered, batch_size=32, shuffle=True)

    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    # training code
    embeddings_train = create_embedding_matrix(index_map, glove_embeddings, 100)

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
