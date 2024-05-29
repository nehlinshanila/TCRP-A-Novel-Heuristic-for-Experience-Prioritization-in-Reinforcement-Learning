import tensorflow as tf


HUBER_LOSS_DELTA = 2.0


def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = tf.abs(err) < HUBER_LOSS_DELTA

    squared_loss = 0.5 * tf.square(err)
    # quadratic_loss = HUBER_LOSS_DELTA * (tf.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    linear = HUBER_LOSS_DELTA * (tf.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, squared_loss, linear)  # Keras does not cover where function in tensorflow :-(
    loss = tf.reduce_mean(loss)
    print(f'Loss : {loss}')

    return loss
