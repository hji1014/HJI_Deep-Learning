""" [TF/Keras] """

""" Luong Attention """
import tensorflow as tf
class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.Wa = tf.keras.layers.Dense(units)

    def call(self, query, values):
        # query: 디코더의 은닉 상태 (batch_size, hidden_size)
        # values: 인코더의 출력 시퀀스 (batch_size, max_len, hidden_size)

        query_with_time_axis = tf.expand_dims(query, 1)     # (32, 1, 128)

        # 점수 계산
        score = tf.matmul(self.Wa(query_with_time_axis), values, transpose_b=True)  # (32, 1, 10)
        attention_weights = tf.nn.softmax(score, axis=1)    # (32, 1, 10)

        # 가중치를 적용한 값의 합 (32, 1, 128)
        context_vector = attention_weights @ values     # context_vector = tf.matmul(attention_weights, values)와 같음

        return context_vector, attention_weights

# Usage example
units = 128
attention_layer = LuongAttention(units)
query = tf.random.normal((32, units))
values = tf.random.normal((32, 10, units))

context_vector, attention_weights = attention_layer(query, values)


""" Bahdanau Attention """
import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.Wq = tf.keras.layers.Dense(units)
        self.Wv = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query: Hidden state from the decoder (batch_size, hidden_size)
        # values: Encoder output sequence (batch_size, max_len, hidden_size)

        query_with_time_axis = tf.expand_dims(query, 1)

        # Score calculation
        score = self.V(tf.nn.tanh(self.Wq(query_with_time_axis) + self.Wv(values)))
        attention_weights = tf.nn.softmax(score, axis=1)        # (32, 10, 1)

        # Weighted sum of values
        context_vector = tf.matmul(attention_weights, values, transpose_a=True)

        return context_vector, attention_weights

# Usage example
units = 128
attention_layer = BahdanauAttention(units)
query = tf.random.normal((32, units))
values = tf.random.normal((32, 10, units))

context_vector, attention_weights = attention_layer(query, values)