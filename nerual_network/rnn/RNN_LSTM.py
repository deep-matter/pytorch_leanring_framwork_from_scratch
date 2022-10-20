import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import *

# intialze the model


class RNN_(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_, self).__init__()

        self.hidden_size = hidden_size
        self.Hidden_0 = nn.Linear(input_size + hidden_size, hidden_size)
        self.Hidden_1 = nn.Linear(input_size + hidden_size, output_size)
        # [1.57] ===> we want ot hAVE SIZE OF TENSOR.SIZE[57]
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_stat):
        combined_state = torch.cat((input_tensor, hidden_stat), 1)

        hidden_0 = self.Hidden_0(combined_state)
        output_state = self.Hidden_1(combined_state)
        output_state = self.activation(output_state)

        return output_state, hidden_0

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# test the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

category_lines, all_categories = loading_data()
n_categories = len(all_categories)

# One step forward  testing
n_hidden = 128
# input_tensor = line_to_tensor("youness")
# hidden_tensor = torch.zeros(1, n_hidden)
# model = RNN_(num_letters,hidden_tensor,n_categories)
# output, next_hidden = model(input_tensor[0], hidden_tensor)
# print(output.size())

n_hidden = 128
model = RNN_(num_letters, n_hidden, n_categories)

# one step
input_tensor = letter_to_tensor('A')
hidden_tensor = model.init_hidden()

output, next_hidden = model(input_tensor, hidden_tensor)


# whole sequence/name
input_tensor = line_to_tensor('Albert')
hidden_tensor = model.init_hidden()

output, next_hidden = model(input_tensor[0], hidden_tensor)


#############

# function to indexing to which the name belong to conutry


def categorize_to_index(output):
    category_idx = torch.argmax(output).item()

    return all_categories[category_idx]


# print(categorize_to_index(output))


# intialize the model training loop
# hyperparamters
learnin_rate = 0.001
Epochs = 3
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learnin_rate)

def train(line_tensor, category_tensor):
    hidden = model.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
        
    loss = criterion(output, category_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return output, loss.item()

# def Train(line_tensor, category_names):
#     if torch.cuda.is_available():
#         hidden_init = model.init_hidden()
#         for i in range(line_tensor.size()[0]):

#             output, hidden = model(line_tensor[i], hidden_init)
#             loss = criterion(output, category_names)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         return output , loss.item()


current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000

for i in range(n_iters):
    category , line , category_tensor , line_tensor = random_trainig_samples(category_lines,all_categories)
    output, loss = train(line_tensor, category_tensor)

    current_loss += loss 

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
        
    if (i+1) % print_steps == 0:
        guess = categorize_to_index(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")
        
    
plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        
        hidden = rnn.init_hidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = categorize_to_index(output)
        print(guess)


while True:
    sentence = input("Input:")
    if sentence == "quit":
        break
    
    predict(sentence)
