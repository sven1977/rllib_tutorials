import tensorflow as tf


class Model(tf.keras.Model):
    def _init__(self,
                input_space,
                action_space,
                num_outputs,
                name="",
                *,
                layers = (256, 256)):
        super().__init__(name=name)

        self.dense_layers = []
        for i, layer_size in enumerate(layers):
            self.dense_layers.append(tf.keras.layers.Dense(
                layer_size, activation=tf.nn.relu, name=f"dense_{i}"))

        self.logits = tf.keras.layers.Dense(
            num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")
        self.values = tf.keras.layers.Dense(
            1, activation=None, name="values")

    def call(self, inputs, training=None, mask=None):
        out = inputs["obs"]
        for l in self.dense_layers:
            out = l(out)
        logits = self.logits(out)
        values = self.values(out)
        return logits, [], {"vf_preds": tf.reshape(values, [-1])}
