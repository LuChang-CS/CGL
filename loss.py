import tensorflow as tf


def medical_codes_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=1))
