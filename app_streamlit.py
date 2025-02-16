import streamlit as st
import torch
import torch.nn as nn
from utils import ALL_LETTERS, N_LETTERS, load_data, line_to_tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Load model and data
category_lines, all_categories = load_data()
n_categories = len(all_categories)
n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)
checkpoint_path = "rnn_model_200000.pth"
rnn.load_state_dict(torch.load(checkpoint_path))
rnn.eval()

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

def predict(input_line):
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        return category_from_output(output)

# Streamlit UI
st.title("RNN Name Classifier")
st.write("Enter a name, and the model will predict its category.")

name_input = st.text_input("Enter a name:")
if name_input:
    prediction = predict(name_input)
    st.write(f"**Predicted Category:** {prediction}")
