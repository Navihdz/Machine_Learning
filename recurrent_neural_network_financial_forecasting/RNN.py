import torch
import torch.nn as nn

import time
import math
#timeSince import

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sigmoid(self.in2hidden(combined))
        output = self.in2output(hidden)
        return output, hidden
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))
    
#trainig
def train(hidden_size,input_tensor, target_tensor, criterion, optimizer):
    input_size = input_tensor.size(1)
    output_size = target_tensor.size(1)
    model= RNN(input_size, hidden_size, output_size)
    hidden = model.init_hidden()
    optimizer.zero_grad()
    loss = 0
    for i in range(input_tensor.size(0)):
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i].unsqueeze(0))
    loss.backward()
    optimizer.step()
    return output, loss.item() / input_tensor.size(0)

#training
def trainIters(hidden_size, epochs, criterion, optimizer,):
    start = time.time()
    print_every = 1000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every plot_every iters
    for epoch in range(1, epochs + 1):
        output, loss = train(hidden_size, input_tensor, target_tensor, criterion, optimizer)
        total_loss += loss
        if epoch % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), epoch, epoch / epochs * 100, loss))
        if epoch % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
    return all_losses
