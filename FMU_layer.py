"""
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                     *
 * FMU custom layers                                                   *
 *                                                                     *
 *  @authors: Matteo Larcher                                           *
 *                                                                     *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# import libraries
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import tensorflow as tf
import queue

# custom libraries
from xy_FMU import xy_FMU_class

@tf.keras.utils.register_keras_serializable(package="Custom", name="FMUCell")
class FMUCell(tf.keras.layers.Layer):

    def __init__(self, fmu_path, start_time, start_values, parameters,
                 step_size, **kwargs):
        super(FMUCell, self).__init__(**kwargs)

        # instantiate the class
        self.fmu_model = xy_FMU_class(
            fmu_path,
            start_time=start_time,
            start_values=start_values,
            parameters=parameters,
            instance_name=self.name,
        )

        self.start_time = start_time
        self.time = start_time
        self.dt = step_size
        self.state_size = 1  # FIXME: retrieve the output shape from the FMU
        self.gradient_queue = queue.LifoQueue()

    def call(self, input_tensor, states):
        fmu_out = self.fmu_op(input_tensor)
        return fmu_out, [fmu_out]

    @tf.custom_gradient
    @tf.autograph.experimental.do_not_convert
    def fmu_op(self, inputs):

        def fmu_step(inputs):
            self.fmu_model.set_inputs(*tf.unstack(tf.reshape(inputs, [-1])))
            self.fmu_model.do_step(self.time, self.dt)
            self.time += self.dt
            # push the gradient in the queue
            self.gradient_queue.put(
                tf.convert_to_tensor(
                    self.fmu_model.get_directional_derivative(),
                    dtype=tf.keras.backend.floatx()))
            return self.fmu_model.get_outputs()

        outputs = tf.py_function(fmu_step, [inputs], tf.keras.backend.floatx())

        def custom_grad(upstream):

            def grad_step(upstream):
                # pop the gradient from the queue
                return upstream * self.gradient_queue.get()

            grad = tf.py_function(grad_step, [upstream],
                                  tf.keras.backend.floatx())
            return tf.reshape(
                grad, [1, 2])  #FIXME: reshape the gradient (state_size)

        return tf.reshape(
            outputs, [1, 1]
        ), custom_grad  #FIXME: reshape the output and the gradient (state_size)

    def reset_states(self):
        # reset time
        self.time = self.start_time
        # empty queue
        self.gradient_queue.empty()
        #reset FMU states
        self.fmu_model.reset_FMU()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "state_size": self.state_size,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom", name="FMULayer")
class FMULayer(tf.keras.layers.Layer):

    def __init__(self, fmu_path, start_time, start_values, parameters,
                 step_size, **kwargs):
        super(FMULayer, self).__init__()
        self.units = 1
        self.cell = FMUCell(fmu_path, start_time, start_values, parameters,
                            step_size)
        # set initial states
        self.initial_state = tf.constant(
            [[0]],
            dtype=tf.keras.backend.floatx())  #FIXME: set the initial state

        # Use the RNN layer with the custom cell
        self.rnn_layer = tf.keras.layers.RNN(self.cell, **kwargs)

    def build(self, input_shape):
        self.cell.build(input_shape)
        self.built = True

    def call(self, inputs, initial_state=None):
        self.cell.reset_states()
        tf.print("Queue length: ", self.cell.gradient_queue.qsize())
        return self.rnn_layer(inputs, initial_state)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.cell.state_size)
