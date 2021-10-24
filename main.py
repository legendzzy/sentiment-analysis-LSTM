# -*- coding: utf-8 -*-
import torch
from torchtext.legacy import data
import spacy
import dill
import torch.nn as nn

spacy_en = spacy.load('en_core_web_sm')
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

nlp = spacy.load('en_core_web_sm')


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]
        x = self.fc(hidden)
        # x = x / torch.norm(x)
        return x


def predict_sentiment(model, sentence):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    tokenized = tokenizer(sentence)
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_vocab_path = './text_vocab'
    label_vocab_path = './label_vocab'
    TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer,
                      include_lengths=True)  # need to let the RNN knows how long is the sequence
    LABEL = data.Field(dtype=torch.float, sequential=False, use_vocab=False)
    with open(text_vocab_path, 'rb')as f:
        TEXT.vocab = dill.load(f)
    with open(label_vocab_path, 'rb')as f:
        LABEL.vocab = dill.load(f)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX)
    model_path = 'LSTM-model-10.pt'
    trained_model = torch.load(model_path)
    model.load_state_dict(trained_model)
    model.to(device)

    # input sentence
    while(True):
        input_sentence = input("please input a sentence:(input 'exit' to end ) ")
        if input_sentence == 'exit':
            exit(-1)
        if isinstance(input_sentence,str):
            res = predict_sentiment(model, input_sentence)
            if(res>=0.5):
                print("confident score: %f"%res)
                print("positive")
            else:
                print("confident score: %f"%res)
                print("negative")
        else:
            print("input type error!")
            exit(-1)

