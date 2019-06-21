from torch import nn
import torch

# Convention: Features are assumed to come out of a linear layer with shape (batch_size, num_points, num_ch).
# Ray directions, signed distances, and points on lines are assumed to be of images shape, i.e. (batch_size, ch, h, w).

def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1/2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


class RecurrentSignedDistancePointPredictor(nn.Module):
    def __init__(self, in_ch, rnn_type='lstm'):
        super().__init__()

        hidden_size = 16
        self.in_ch = in_ch
        self.rnn_type = rnn_type

        self.rnn = nn.LSTMCell(input_size=in_ch,
                               hidden_size=hidden_size)

        self.rnn.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.rnn)

        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, features, line_direction, point_on_line, prev_state=None):
        batch_size, num_points, num_feats = features.shape

        state = self.rnn(features.view(-1, num_feats), prev_state)

        if state[0].requires_grad:
            state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

        signed_distance = self.out_layer(state[0]).view(batch_size, num_points, 1)

        pred_points = line_from_signed_distance_to_point(signed_distance, line_direction, point_on_line)

        return pred_points, state, signed_distance


def line_from_signed_distance_to_point(distance, line_direction, point_on_line):
    return point_on_line + line_direction * distance



