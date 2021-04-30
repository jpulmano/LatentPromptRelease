import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers


class BidirectionalRNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

    To keep things modular, both question & response sequences are parameters, but only response
    features are used.

    Input:
        num_channels = number of latent categories
    """
    def __init__(
      self, conversation_length, num_channels, embedding_size, num_hidden_layers, regularization_coefficient, pos_weight):

        num_classes = 2

        # Placeholders for input, output and dropout
        self.input_prompts = tf.placeholder(tf.float32, [None, conversation_length, embedding_size], name="input_prompts")
        self.input_responses = tf.placeholder(tf.float32, [None, conversation_length, embedding_size], name="input_responses")

        # Mask for varying number of h vectors (varying number of prompts/responses)
        self.input_masks = tf.placeholder(tf.float32, [None, conversation_length])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")        


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        with tf.variable_scope("classifier"):            
            self.batch_size = tf.shape(self.input_prompts)[0]
            attention_dims = 100

            # (batch_size * max_len * embedding)
            single_batch = tf.reshape(self.input_prompts, shape=[self.batch_size * conversation_length, int(embedding_size)])

            # Category inference layer weights/bias
            self.W_q = tf.Variable(tf.truncated_normal([embedding_size, num_channels]), name="W_q")
            self.b_q = tf.Variable(tf.constant(0.1, shape=[num_channels]), name="b_q")
            
            # Linear layer w/ dropout
            query = tf.nn.xw_plus_b(tf.nn.dropout(single_batch, keep_prob=self.dropout_keep
                ), self.W_q, self.b_q, name="query_layer")

            # Latent category membership (h_{i1})
            query_3D = tf.reshape(query, [self.batch_size, conversation_length, num_channels])
            self.usefulness = tf.math.softmax(query_3D, axis=2, name="channel_saliences")
            
            # Specifically for BERT (score of each of K prompt categories on all h vectors) 
            self.usefulness = tf.clip_by_value(self.usefulness, tf.constant(1e-15), tf.constant(1.0))

            usefulness_dropout = tf.nn.dropout(self.usefulness, keep_prob=1.0) # (batch x max_conversation_length x channel_number)

            # h - category membership
            # r - response vectors
            # at every step, compute h * r (category membership * response vector)
            # feed h * r into RNN
            # once input is exhausted, take final hidden layer and throw it into a decision layer

            # loop through every channel
            # loop through every prompt/response pair
            
            # Initialize LSTM (TODO: move this up later)

            # self.LSTM = keras.Sequential()
            # self.LSTM.add(layers.LSTM(2 * embedding_size))
            self.LSTM = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(2 * embedding_size, dropout = 0.2),
                merge_mode='sum'
            )

            channel_evidence = []
            for channel in range(num_channels):

                # Calculate h * r
                h_r = usefulness_dropout[:, :, channel, tf.newaxis] * tf.concat([self.input_prompts, self.input_responses], axis=2)

                # Get last hidden state from LSTM
                output = self.LSTM(h_r)

                channel_evidence.append(output)

            # Create a huge vector out of the "Category-Aware Response Aggregations" to pass into a decision layer
            combined_evidence = tf.concat(axis=1, values=channel_evidence)

            # Prompts and responses [batch x (2 * embedding_size * prompt categories)]
            combined_evidence = tf.reshape(combined_evidence, [self.batch_size, embedding_size * 2 * num_channels])

            # Throw into a decision layer
            if num_hidden_layers > 0:
                combined_evidence = tf.layers.dense(
                inputs = tf.nn.dropout(combined_evidence, keep_prob=self.dropout_keep),
                units = embedding_size,
                name="scores_hidden_layer",
                activation = tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=43),
            )   

            # Get scores and predictions
            self.scores = tf.layers.dense(
                inputs = tf.nn.dropout(combined_evidence, keep_prob=self.dropout_keep),
                units = 2,
                name="scores_layer",
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42),
            )                  
            self.predictions = tf.argmax(self.scores, 1, name="compute_predictions")
               
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y) # Not weighted
            losses = tf.nn.weighted_cross_entropy_with_logits(self.input_y, self.scores, pos_weight)
            
            norm = 1.0 / tf.reduce_sum(tf.reduce_sum(tf.cast(self.input_masks[:, :], dtype=tf.float32), axis=1), axis=0)
            
            entropy_component = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(usefulness_dropout * tf.math.log(usefulness_dropout), axis=2) * 
                tf.cast(self.input_masks[:, :], dtype=tf.float32), axis=1), axis=0)      
            

            self.loss = tf.reduce_mean(losses) - regularization_coefficient * norm * entropy_component

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")