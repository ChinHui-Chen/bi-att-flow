"Define a PTR cell."
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell_impl import _linear

class PTRCell(tf.contrib.rnn.RNNCell):

    def __init__(self, nSymbols = 100, nRoles = 10, dSymbols = 10, dRoles = 10, dEmb = 100, kernel_initializer=None, recurrent_initializer=None, bias_initializer=None, batch_size = 32, reuse=None):
        super(PTRCell, self).__init__(_reuse=reuse)
        self._nSymbols = nSymbols
        self._nRoles = nRoles
        self._dSymbols = dSymbols
        self._dRoles = dRoles
        self._dEmb = dEmb
        self._num_units = self._dSymbols * self._dRoles
        self._kernel_initializer = kernel_initializer
        self._recurrent_initializer = recurrent_initializer
        self._bias_initializer = bias_initializer
        self._batch_size = batch_size

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

#    def zero_state(self, batch_size, dtype):
#        return tf.zeros([batch_size, self._dSymbols * self._dRoles], dtype=dtype)

    def call(self, inputs, state):
        print("state:")
        print(state.get_shape())
        h = state

        with vs.variable_scope("att_ast"):
            a_s_t = math_ops.sigmoid(
                _linear([inputs, h], self._nSymbols, True, self._bias_initializer,
                        self._kernel_initializer))
            # a_s_t.shape = (batch_size, nSymbols)
            #              x (nSymbol,dSymbol) = (batch_size, dSymbol)                

        with vs.variable_scope("att_art"):
            a_r_t = math_ops.sigmoid(
                _linear([inputs, h], self._nRoles, True, self._bias_initializer,
                        self._kernel_initializer))
            # a_r_t.shape = (batch_size, nRoles)
            #              x (nRoles, dRoles) = (batch_size, dRoles)
        with vs.variable_scope("SR"):
            S = tf.get_variable('S', [self._nSymbols, self._dSymbols], initializer=self._recurrent_initializer)
            R = tf.get_variable('R', [self._nRoles, self._dRoles], initializer=self._recurrent_initializer)

        S_matmul = tf.matmul(a_s_t, S)
        S_matmul = tf.reshape( S_matmul, [-1, self._dSymbols, 1] )
        R_matmul = tf.matmul(a_r_t, R)
        R_matmul = tf.reshape( R_matmul, [-1, self._dRoles, 1] )
        R_matmul = tf.transpose( R_matmul, perm=[0, 2, 1] )

        output = tf.matmul( S_matmul, R_matmul )
        output = tf.reshape(output, [-1, self._dSymbols*self._dRoles])

        #return output, tf.contrib.rnn.LSTMStateTuple(c, output)
        return output, output

        # inputs.shape = (batch_size, dEmb)
        # state.shape = (batch_size, state_size = output_size = 10x10)

        # v_t = ( S a_s_t ) ( R a_r_t )^T = S(a_s_t a_r_t^T)R^T
        # a_s_t = sigmoid( (W_s_in w_t) + (W_s_rec v_t-1) + b_s )
        # a_r_t = sigmoid( (W_r_in w_t) + (W_r_rec v_t-1) + b_r ) 
        # v_t.shape = (dSymbols, dRoles)
        # v_t-1.shape = (dSymbols*dRoles, 1)
        # S.shape = (dSymbols, nSymbols)
        # R.shape = (dRoles, nRoles)
        # a_s_t.shape = (nSymbols, 1)
        # a_r_t.shape = (nRoles, 1)
        # w_t.shape = (dEmb, 1)
        # W_s_in.shape = (nSymbols, dEmb)
        # W_s_rec.shape = (nSymbols, dSymbols*dRoles)
        # b_s.shape = (nSymbols, 1)
        # W_r_in.shape = (nRoles, dEmb)
        # W_r_rec.shape = (nRoles, dSymbols*dRoles) 
        # b_r.shape = (nRoles, 1)
        #batch = tf.shape(inputs)[0]
        #print("batch")
        #print(batch)

####        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):

#            S = tf.get_variable('S', [self._batch_size, self._dSymbols, self._nSymbols], initializer=self._recurrent_initializer)
#            R = tf.get_variable('R', [self._batch_size, self._dRoles, self._nRoles], initializer=self._recurrent_initializer)

            # TODO: input only 800?
#            inputs = tf.reshape( inputs, [-1, self._dEmb, 1] )
#            print('state')
#            print(state.get_shape())
#            state = tf.reshape( state, [-1, self._dSymbols*self._dRoles, 1] ) 

#            W_s_in = tf.get_variable('W_s_in', [self._batch_size, self._nSymbols, self._dEmb])
#            W_s_rec = tf.get_variable('W_s_rec', [self._batch_size, self._nSymbols, self._dSymbols*self._dRoles])
#            b_s = tf.get_variable('b_s', [self._batch_size, self._nSymbols, 1])
###            W_s_in = tf.get_variable('W_s_in', [self._dEmb, self._nSymbols])
###            W_s_rec = tf.get_variable('W_s_rec', [self._dSymbols*self._dRoles, self._nSymbols])
###            b_s = tf.get_variable('b_s', [self._nSymbols])


#            W_r_in = tf.get_variable('W_r_in', [self._batch_size, self._nRoles, self._dEmb])
#            W_r_rec = tf.get_variable('W_r_rec', [self._batch_size, self._nRoles, self._dSymbols*self._dRoles])
#            b_r = tf.get_variable('b_r', [self._batch_size, self._nRoles, 1])
###            W_r_in = tf.get_variable('W_r_in', [self._dEmb, self._nRoles])
###            W_r_rec = tf.get_variable('W_r_rec', [self._dSymbols*self._dRoles, self._nRoles])
###            b_r = tf.get_variable('b_r', [self._nRoles])

            # compute attention
#            a_s_t = tf.sigmoid( tf.contrib.rnn.basicRNNCell._linear([inputs, state], self._nSymbols, True, scope="a_s_t") ) # shape = [batch x nSymbols]
###            a_s_t = tf.sigmoid( tf.matmul(inputs, W_s_in) + tf.matmul(state, W_s_rec) + b_s )
#            a_s_t = tf.reshape(a_s_t, [-1, 1, self._nSymbols])

#            a_r_t = tf.sigmoid( _linear([inputs, state], self._nRoles, True, scope="a_r_t") )   # shape = [batch x nRoles]
###            a_r_t = tf.sigmoid( tf.matmul(inputs, W_r_in) + tf.matmul(state, W_r_rec) + b_r )
#            a_r_t = tf.reshape(a_r_t, [-1, 1, self._nRoles])

            # compute TPR
###            S = tf.get_variable('S', [self._nSymbols, self._dSymbols], initializer=self._recurrent_initializer)
###            R = tf.get_variable('R', [self._nRoles, self._dRoles], initializer=self._recurrent_initializer)

            # a_s_t.shape = (batch_size, 1, nSymbol)
            # S.shape = (nSymbol, dSymbol)
            # matmul = (batch_size, dSymbol)

            # a_r_t.shape = (batch_size, nRole, 1)
            # R
            # matmul = (batch_size, dRole)
###            S_matmul = tf.matmul(a_s_t, S)
###            S_matmul = tf.reshape( S_matmul, [-1, self._dSymbols, 1] )
###            R_matmul = tf.matmul(a_r_t, R)
###            R_matmul = tf.reshape( R_matmul, [-1, 1, self._dRoles] )
###            output = tf.matmul( S_matmul, R_matmul )
###            output = tf.reshape(output, [-1, self._dSymbols*self._dRoles])
###            # output.shape goal: (batch_size, 10x10)
###            
###            return output, output
#            next_states = []
#            for j, state_j in enumerate(state): # Hidden State (j)
#                key_j = tf.expand_dims(self._keys[j], axis=0)
#                gate_j = self.get_gate(state_j, key_j, inputs)
#                candidate_j = self.get_candidate(state_j, key_j, inputs, U, V, W, U_bias)
#
#                # Equation 4: h_j <- h_j + g_j * h_j^~
#                # Perform an update of the hidden state (memory).
#                state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j
#
#                # Equation 5: h_j <- h_j / \norm{h_j}
#                # Forget previous memories by normalization.
#                state_j_next_norm = tf.norm(
#                    tensor=state_j_next,
#                    ord='euclidean',
#                    axis=-1,
#                    keep_dims=True)
#                state_j_next_norm = tf.where(
#                    tf.greater(state_j_next_norm, 0.0),
#                    state_j_next_norm,
#                    tf.ones_like(state_j_next_norm))
#                state_j_next = state_j_next / state_j_next_norm
#
#                next_states.append(state_j_next)
#            state_next = tf.concat(next_states, axis=1)
#        return state_next, state_next


