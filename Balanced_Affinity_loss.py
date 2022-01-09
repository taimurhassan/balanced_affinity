import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow.keras.backend as K
import torch
import torch.nn.functional as F

class ClusteringAffinity(layers.Layer):
    def __init__(self, n_classes, n_centers, sigma, **kwargs):
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.sigma = sigma
        super().__init__(**kwargs)

    def build(self, input_shape):
        # batch_size = N, n_classes = C, n_latent_dims = h, centers = m
        # center(W) = (C, m, h)
        # input(f) = (N, h)
        self.W = self.add_weight(name="W", shape=(self.n_classes, self.n_centers, input_shape[1]),
                                 initializer="he_normal", trainable=True)

        super().build(input_shape)

    def upper_triangle(self, matrix):
        upper = tf.compat.v1.matrix_band_part(matrix, 0, -1)
        diagonal = tf.compat.v1.matrix_band_part(matrix, 0, 0)
        diagonal_mask = tf.sign(tf.abs(tf.compat.v1.matrix_band_part(diagonal, 0, 0)))
        return upper * (1.0 - diagonal_mask)

    def call(self, f):
        # Euclidean space similarity measure
        # calculate d(f_i, w_j)
        f_expand = tf.expand_dims(tf.expand_dims(f, axis=1), axis=1)
        w_expand = tf.expand_dims(self.W, axis=0)
        fw_norm = tf.reduce_sum((f_expand-w_expand)**2, axis=-1)
        distance = tf.exp(-fw_norm/self.sigma)
        distance = tf.reduce_max(distance, axis=-1) # (N,C,m)->(N,C)

        # Regularization
        hidden_layers = K.int_shape(self.W)[2]
        mc = self.n_classes * self.n_centers
        w_reshape = tf.reshape(self.W, [mc, hidden_layers])
        w_reshape_expand1 = tf.expand_dims(w_reshape, axis=0)
        w_reshape_expand2 = tf.expand_dims(w_reshape, axis=1)
        w_norm_mat = tf.reduce_sum((w_reshape_expand2 - w_reshape_expand1)**2, axis=-1)
        w_norm_upper = self.upper_triangle(w_norm_mat)
        mu = 2.0 / (mc**2 - mc) * tf.reduce_sum(w_norm_upper)
        residuals = self.upper_triangle((w_norm_upper - mu)**2)
        rw = 2.0 / (mc**2 - mc) * tf.reduce_sum(residuals)

        batch_size = tf.shape(f)[0]
        rw_broadcast = tf.ones((batch_size,1)) * rw

        # outputs distance(N, C) + rw(N,)
        output = tf.concat([distance, rw_broadcast], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        # ! important !
        # To calculate the regularization term, output n_dimensions is 1 more.
        # Please ignore it at predict time
        return (input_shape[0], self.n_classes+1)

def balanced_affinity_loss(lambd):
    def loss(y_true_plusone, y_pred_plusone):
        # true
        no_of_classes = 2
        labels = torch.randint(0, no_of_classes, size=(32,))
        beta = 0.9

        # samples_per_cls = [80957, 3131, 1943, 2199, 3961, 983] #SIXray10
        # samples_per_cls = [873872, 3131, 1943, 2199, 3961, 983]  # SIXray100
        # samples_per_cls = [1993,1044,1863,1978,2042] #OPIXray
        #samples_per_cls = [80957, 8929] #SIXray100
        # samples_per_cls = [11979, 1433]  # SIXray100
        # samples_per_cls = [4609, 1050301]  # SIXray1000
        samples_per_cls = [1548, 10020] #COMPASS-XP
        # samples_per_cls = [1953, 9971]
        # samples_per_cls = [3699, 8225]
        # samples_per_cls = [995, 9934]
        # samples_per_cls = [2204, 9720]
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        labels_one_hot = F.one_hot(labels, no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)
        weights = weights.numpy()
        weights = tf.convert_to_tensor(weights)

        onehot = y_true_plusone[:, :-1]
        # pred
        distance = y_pred_plusone[:, :-1]
        rw = tf.reduce_mean(y_pred_plusone[:, -1])
        # L_mm
        d_fi_wyi = tf.reduce_sum(onehot * distance, axis=-1, keepdims=True)
        losses = tf.maximum(lambd + distance - d_fi_wyi, 0.0)
        L_mm = tf.reduce_sum(losses * (1.0-onehot), axis=-1) # j!=y_i
        # L_mm = tf.reduce_mean(L_mm)
        #weights = ((1 - 0.99) / (1 - (0.99) ** 2))
        weights = tf.reduce_sum(weights).numpy()
        weights /= tf.reduce_sum(labels).numpy()
        weighted_loss = (weights * (L_mm + rw))

        return weighted_loss
    return loss

