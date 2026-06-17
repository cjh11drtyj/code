import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers


class Multi_Head_Attention(Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.initializer = initializers.get(initializer)

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        self.d_model = d_model

        self.q_weight = self.add_weight(
            shape=(d_model, d_model),
            initializer=self.initializer,
            name="q_weight",
            trainable=True,
        )

        self.k_weight = self.add_weight(
            shape=(d_model, d_model),
            initializer=self.initializer,
            name="k_weight",
            trainable=True,
        )

        self.v_weight = self.add_weight(
            shape=(d_model, d_model),
            initializer=self.initializer,
            name="v_weight",
            trainable=True,
        )

        self.out_weight = self.add_weight(
            shape=(d_model, d_model),
            initializer=self.initializer,
            name="out_weight",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, time_steps, hidden_dim)
        # 例如: (batch_size, 150, 64)

        q = tf.tensordot(inputs, self.q_weight, axes=[[2], [0]])
        k = tf.tensordot(inputs, self.k_weight, axes=[[2], [0]])
        v = tf.tensordot(inputs, self.v_weight, axes=[[2], [0]])

        # score: (batch_size, time_steps, time_steps)
        score = tf.matmul(q, k, transpose_b=True)
        score = score / tf.sqrt(tf.cast(self.d_model, tf.float32))

        # 对每个样本内部的时间步做softmax
        attention_weight = tf.nn.softmax(score, axis=-1)

        # attention_output: (batch_size, time_steps, hidden_dim)
        attention_output = tf.matmul(attention_weight, v)

        # out: (batch_size, time_steps, hidden_dim)
        out = tf.tensordot(attention_output, self.out_weight, axes=[[2], [0]])

        # 取最后一个时间步的注意力输出，用于预测下一时刻误差
        out = out[:, -1, :]

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "initializer": initializers.serialize(self.initializer)
        })
        return config
