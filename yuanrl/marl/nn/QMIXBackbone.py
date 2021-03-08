from torch import nn
from torch.nn import functional as F
import torch

class SingleAgent(nn.Module):
    def __init__(self, kwargs):
        super(SingleAgent, self).__init__()

        self.fc1 = nn.Linear(kwargs['input_dim'], 64)
        self.fc2 = nn.Linear(64, 128)
        self.rnn = nn.GRUCell(128, kwargs['hidden_dim'])
        self.fc3 = nn.Linear(kwargs['hidden_dim'], kwargs['output_dim'])

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, local_state, h_in):
        x = self.leaky_relu(self.fc1(local_state))
        x = self.leaky_relu(self.fc2(x))
        h_out = self.rnn(x, h_in)
        x = self.fc3(h_out)

        return x, h_out
    
class MultiAgent(nn.Module):
    def __init__(self, device, agents_num, agent_kwargs):
        super(MultiAgent, self).__init__()

        self.device = device
        self.all_agents = list()
        for i in range(agents_num):
            self.all_agents.append(SingleAgent(kwargs=agent_kwargs))

    def forward(self, local_state, ph, flag):
        all_local_qs = []
        all_h = []

        for idx, agent in enumerate(self.all_agents):
            if flag == 'eval':
                ls_tensor = torch.FloatTensor(local_state[idx]).unsqueeze(0).to(self.device)
                ph_tensor = torch.FloatTensor(ph[idx]).unsqueeze(0).to(self.device)
            else:
                ''' train '''
                ls_tensor = torch.FloatTensor(local_state[:, idx, :]).to(self.device)
                ph_tensor = torch.FloatTensor(ph[:, idx, :]).to(self.device)
            local_qs, h = agent(ls_tensor, ph_tensor)
            all_local_qs.append(local_qs)
            all_h.append(h)

        all_local_qs = torch.stack(all_local_qs, dim=1)
        all_h = torch.stack(all_h, dim=1)

        return all_local_qs, all_h

class MixingNet(nn.Module):
    def __init__(self, kwargs):
        super(MixingNet, self).__init__()
        self.hyper_w1 = nn.Linear(kwargs['hyper_input_dim'], kwargs['mixing_input_dim'] * 64)
        self.hyper_b1 = nn.Linear(kwargs['hyper_input_dim'], 64)
        self.hyper_w2 = nn.Linear(kwargs['hyper_input_dim'], 64 * 128)
        self.hyper_b2 = nn.Linear(kwargs['hyper_input_dim'], 128)
        self.hyper_w3 = nn.Linear(kwargs['hyper_input_dim'], 128 * kwargs['mixing_output_dim'])
        self.hyper_b3 = nn.Sequential(
            nn.Linear(kwargs['hyper_input_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, kwargs['mixing_output_dim'])
        )

        self.elu = nn.ELU()

    def forward(self, global_state, q_values):
        w1 = self.hyper_w1(global_state)
        w1 = torch.abs(w1.view(-1, q_values.shape[2], 64))
        # print(w1.shape)
        b1 = self.hyper_b1(global_state)
        b1 = b1.view(-1, 1, 64)
        # print(b1.shape)

        w2 = self.hyper_w2(global_state)
        w2 = torch.abs(w2.view(-1, 64, 128))
        # print(w2.shape)
        b2 = self.hyper_b2(global_state)
        b2 = b2.view(-1, 1, 128)
        # print(b2.shape)

        w3 = self.hyper_w3(global_state)
        w3 = torch.abs(w3.view(-1, 128, 1))
        # print(w3.shape)
        b3 = self.hyper_b3(global_state)
        b3 = b3.view(-1, 1, 1)
        # print(b3.shape)

        x = self.elu(torch.bmm(q_values, w1) + b1)
        x = self.elu(torch.bmm(x ,w2) + b2)
        x = torch.bmm(x, w3) + b3

        return x[:, 0, :]