""" [Pytorch] """

""" Luong Attention """
import torch
import torch.nn as nn
class LuongAttention(nn.Module):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.Wa = nn.Linear(units, units)

    def forward(self, query, values):
        # query: 디코더의 은닉 상태 (batch_size, hidden_size)
        # values: 인코더의 출력 시퀀스 (batch_size, max_len, hidden_size)

        query_with_time_axis = query.unsqueeze(1)

        # 점수 계산
        score = torch.matmul(self.Wa(query_with_time_axis), values.transpose(1, 2))
        attention_weights = torch.nn.functional.softmax(score, dim=1)

        # 가중치를 적용한 값의 합
        context_vector = torch.matmul(attention_weights, values)        # context_vector = attention_weights @ values

        return context_vector, attention_weights

# Usage example
units = 128
attention_layer = LuongAttention(units)
query = torch.rand((32, units))
values = torch.rand((32, 10, units))

context_vector, attention_weights = attention_layer(query, values)


""" Bahdanau Attention """
import torch
import torch.nn as nn
class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.Wq = nn.Linear(units, units)
        self.Wv = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, query, values):
        # query: 디코더의 은닉 상태 (batch_size, hidden_size)
        # values: 인코더의 출력 시퀀스 (batch_size, max_len, hidden_size)

        query_with_time_axis = query.unsqueeze(1)

        # 점수 계산
        score = self.V(torch.tanh(self.Wq(query_with_time_axis) + self.Wv(values)))
        attention_weights = torch.nn.functional.softmax(score, dim=1)

        # 가중치를 적용한 값의 합
        context_vector = torch.matmul(attention_weights.transpose(1, 2), values)

        return context_vector, attention_weights

# Usage example
units = 128
attention_layer = BahdanauAttention(units)
query = torch.rand((32, units))
values = torch.rand((32, 10, units))

context_vector, attention_weights = attention_layer(query, values)
