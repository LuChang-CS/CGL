import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, GRUCell
from tensorflow.keras.layers import Concatenate, Activation, RNN, StackedRNNCells
from tensorflow.keras.initializers import GlorotUniform


class HierarchicalEmbedding(Layer):
    def __init__(self, code_levels, code_num_in_levels, code_dims, name='hierarchical_embedding'):
        super().__init__(name=name)
        self.level_num = len(code_num_in_levels)
        self.code_levels = code_levels  # (leaf code num * level_num)
        self.level_embeddings = [self.add_weight(name='hier_emb_level_%d' % level,
                                                 shape=(code_num, code_dim),
                                                 initializer=GlorotUniform(),
                                                 trainable=True)
                                 for level, (code_num, code_dim) in enumerate(zip(code_num_in_levels, code_dims))]

    def call(self, inputs=None):
        """
            return: (code_num, embedding_size)
        """
        embeddings = [tf.nn.embedding_lookup(self.level_embeddings[level], self.code_levels[:, level])
                      for level in range(self.level_num)]
        embeddings = Concatenate()(embeddings)
        return embeddings


class PatientEmbedding(Layer):
    def __init__(self, patient_num, patient_dim, name='patient_embedding'):
        super().__init__(name=name)
        self.patient_embeddings = self.add_weight(name='p_emb',
                                                  shape=(patient_num, patient_dim),
                                                  initializer=GlorotUniform(),
                                                  trainable=True)

    def call(self, inputs=None):
        return self.patient_embeddings


class WordEmbedding(Embedding):
    def __init__(self, word_num, word_dim, embeddings_initializer='glorot_uniform', name='word_embedding'):
        super().__init__(word_num, word_dim, embeddings_initializer=embeddings_initializer, name=name)


class GraphConvBlock(Layer):
    def __init__(self, node_type, dim, adj, name='graph_conv_block'):
        super().__init__(name=name)
        self.node_type = node_type
        self.adj = adj
        self.dense = Dense(dim, activation=None, name=name + '_dense')
        self.activation = Activation('relu', name=name + '_activation')
        self.bn = BatchNormalization(name=name + 'bn')

    def call(self, embedding, embedding_neighbor, weight_decay=None):
        output = embedding + tf.matmul(self.adj, embedding_neighbor)
        if self.node_type == 'code':
            assert weight_decay is not None
            output += tf.matmul(weight_decay, embedding)
        output = self.dense(output)
        output = self.bn(output)
        output = self.activation(output)
        return output


def norm_no_nan(x):
    return tf.math.divide_no_nan(x, tf.reduce_sum(x, axis=-1, keepdims=True))


class GraphConvolution(Layer):
    def __init__(self, patient_dim, code_dim,
                 patient_code_adj, code_code_adj,
                 patient_hidden_dims, code_hidden_dims, name='graph_convolution'):
        super().__init__(name=name)
        self.patient_code_adj = norm_no_nan(patient_code_adj)  # (patient_num, code_num)
        self.code_patient_adj = norm_no_nan(tf.transpose(patient_code_adj))  # (code_num, patient_num)
        self.code_code_adj = code_code_adj  # (code_num, code_num)

        self.patient_blocks = [
            GraphConvBlock('patient', dim, self.patient_code_adj, name='patient_graph_block_%d' % layer)
            for layer, dim in enumerate(patient_hidden_dims)]
        self.code_blocks = [GraphConvBlock('code', dim, self.code_patient_adj, name='code_graph_block_%d' % layer)
                            for layer, dim in enumerate(code_hidden_dims)]

        c2p_dims = ([patient_dim] + patient_hidden_dims)[:-1]
        p2c_dims = ([code_dim] + code_hidden_dims)[:-1]
        self.c2p_denses = [Dense(dim, activation=None, name='code_to_patient_dense_%d' % layer)
                           for layer, dim in enumerate(c2p_dims)]
        self.p2c_denses = [Dense(dim, activation=None, name='patient_to_code_dense_%d' % layer)
                           for layer, dim in enumerate(p2c_dims)]

        code_num = code_code_adj.shape[0]
        self.miu = self.add_weight(name='miu', shape=(code_num,), trainable=True)
        self.theta = self.add_weight(name='theta', shape=(code_num,), trainable=True)

    def call(self, patient_embeddings, code_embeddings):
        weight_decay = tf.nn.sigmoid(self.miu * self.code_code_adj + self.theta)
        weight_decay = norm_no_nan(weight_decay)
        # weight_decay = None
        for c2p_dense, p2c_dense, patient_block, code_block in zip(self.c2p_denses, self.p2c_denses,
                                                                   self.patient_blocks, self.code_blocks):
            code_embeddings_p = c2p_dense(code_embeddings)
            patient_embeddings_new = patient_block(patient_embeddings, code_embeddings_p)
            patient_embeddings_c = p2c_dense(patient_embeddings)
            code_embeddings = code_block(code_embeddings, patient_embeddings_c, weight_decay)
            patient_embeddings = patient_embeddings_new
        patient_embeddings_c = self.p2c_denses[-1](patient_embeddings)
        code_embeddings = self.code_blocks[-1](code_embeddings, patient_embeddings_c, weight_decay)
        return patient_embeddings, code_embeddings


class VisitEmbedding(Layer):
    def __init__(self, max_seq_len, name='visit_embedding'):
        super().__init__(name=name)
        self.max_seq_len = max_seq_len

    def call(self, code_embeddings, visit_codes, visit_lens):
        """
            visit_codes: (batch_size, max_seq_len, max_code_num_in_a_visit)
        """
        visit_codes_embedding = tf.nn.embedding_lookup(code_embeddings, visit_codes)  # (batch_size, max_seq_len, max_code_num_in_a_visit, code_dim)
        visit_codes_mask = tf.expand_dims(visit_codes > 0, axis=-1)
        visit_codes_mask = tf.cast(visit_codes_mask, visit_codes_embedding.dtype)
        visit_codes_embedding *= visit_codes_mask  # (batch_size, max_seq_len, max_code_num_in_a_visit, code_dim)
        visit_codes_num = tf.expand_dims(tf.reduce_sum(tf.cast(visit_codes > 0, visit_codes_embedding.dtype), axis=-1), axis=-1)
        visits_embeddings = tf.math.divide_no_nan(tf.reduce_sum(visit_codes_embedding, axis=-2), visit_codes_num)  # (batch_size, max_seq_len, code_dim)
        visit_mask = tf.expand_dims(tf.sequence_mask(visit_lens, self.max_seq_len, dtype=visits_embeddings.dtype), axis=-1)  # (batch_size, max_seq_len, 1)
        visits_embeddings *= visit_mask  # (batch_size, max_seq_len, code_dim)
        return visits_embeddings


def masked_softmax(inputs, mask):
    inputs = inputs - tf.reduce_max(inputs, keepdims=True, axis=-1)
    exp = tf.exp(inputs) * mask
    result = tf.math.divide_no_nan(exp, tf.reduce_sum(exp, keepdims=True, axis=-1))
    return result


class Attention(Layer):
    def __init__(self, attention_dim, name='attention'):
        super().__init__(name=name)
        self.attention_dim = attention_dim
        self.u_omega = self.add_weight(name=name + '_u', shape=(attention_dim,), initializer=GlorotUniform())
        self.w_omega = None

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.w_omega = self.add_weight(name=self.name + '_w', shape=(hidden_size, self.attention_dim), initializer=GlorotUniform())

    def call(self, x, mask=None):
        """
            x: (batch_size, max_seq_len, rnn_dim[-1] / hidden_size)
        """
        t = tf.matmul(x, self.w_omega)
        vu = tf.tensordot(t, self.u_omega, axes=1)  # (batch_size, max_seq_len)
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = tf.nn.softmax(vu)  # (batch_size, max_seq_len)
        output = tf.reduce_sum(x * tf.expand_dims(alphas, -1), axis=-2)  # (batch_size, rnn_dim[-1] / hidden_size)
        return output, alphas


class TemporalEmbedding(Layer):
    def __init__(self, rnn_dims, attention_dim, max_seq_len, cell_type=GRUCell, name='code_ra'):
        super().__init__(name=name)
        rnn_cells = [cell_type(rnn_dim) for rnn_dim in rnn_dims]
        stacked_rnn = StackedRNNCells(rnn_cells)
        self.rnn_layers = RNN(stacked_rnn, return_sequences=True, name=name + 'rnn')
        self.attention = Attention(attention_dim, name=name + '_attention')
        self.max_seq_len = max_seq_len

    def call(self, embeddings, lens):
        seq_mask = tf.sequence_mask(lens, self.max_seq_len, dtype=embeddings.dtype)
        outputs = self.rnn_layers(embeddings) * tf.expand_dims(seq_mask, axis=-1)  # (batch_size, max_seq_len, rnn_dim[-1])
        outputs, alphas = self.attention(outputs, seq_mask)  # (batch_size, rnn_dim[-1])
        return outputs, alphas


def log_no_nan(x):
    mask = tf.cast(x == 0, dtype=x.dtype)
    x = x + mask
    return tf.math.log(x)


class NoteAttention(Layer):
    def __init__(self, attention_dim, name='attention'):
        super().__init__(name=name)
        self.attention_dim = attention_dim
        self.w_omega = None

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.w_omega = self.add_weight(name=self.name + '_w', shape=(hidden_size, self.attention_dim), initializer=GlorotUniform())

    def call(self, x, ctx_vector, mask=None):
        """
            x: (batch_size, max_seq_len, rnn_dim[-1] / hidden_size)
        """
        t = tf.matmul(x, self.w_omega)
        vu = tf.reduce_sum(t * tf.expand_dims(ctx_vector, axis=1), axis=-1)  # (batch_size, max_seq_len)
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = tf.nn.softmax(vu)  # (batch_size, max_seq_len)
        output = tf.reduce_sum(t * tf.expand_dims(alphas, -1), axis=-2)  # (batch_size, rnn_dim[-1] / hidden_size)
        return output, alphas


class NoteEmbedding(Layer):
    def __init__(self, attention_dim, max_seq_len, lambda_, name='note_embedding'):
        super().__init__(name=name)
        self.attention = NoteAttention(attention_dim, name=name + '_attention')
        self.max_seq_len = max_seq_len
        print(max_seq_len)
        self.lambda_ = lambda_

    def call(self, word_embeddings, word_lens, visit_output, tf_idf=None, training=True):
        word_mask = tf.sequence_mask(word_lens, self.max_seq_len, dtype=word_embeddings.dtype)
        word_embeddings = word_embeddings * tf.expand_dims(word_mask, axis=-1)
        outputs, alphas = self.attention(word_embeddings, visit_output, word_mask)
        if training:
            loss = alphas * log_no_nan(tf_idf) + (1 - alphas) * log_no_nan(1 - tf_idf)
            loss = -self.lambda_ * tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
            self.add_loss(loss)
        return outputs, alphas
