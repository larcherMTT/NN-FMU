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
import ctypes

# custom libraries
from xy_FMU import xy_FMU_class


#  _____ __  __ _   _  ____     _ _
# |  ___|  \/  | | | |/ ___|___| | |
# | |_  | |\/| | | | | |   / _ \ | |
# |  _| | |  | | |_| | |__|  __/ | |
# |_|   |_|  |_|\___/ \____\___|_|_|

@tf.keras.utils.register_keras_serializable(package="Custom", name="FMUCell")
class FMUCell(tf.keras.layers.Layer):

    def __init__(
        self,
        fmu_path,
        start_time,
        start_values,
        parameters,
        step_size,
        do_step_in_gradient=False,
        **kwargs
    ):
        super(FMUCell, self).__init__(**kwargs)

        # instantiate the class
        self.fmu_model = xy_FMU_class(
            fmu_path,
            start_time=start_time,
            start_values=start_values,
            parameters=parameters,
            instance_name=self.name,
        )

        self.do_step_in_gradient = do_step_in_gradient
        self.start_time = start_time
        self.time = start_time
        self.dt = step_size
        self.state_size = 1 # state: pointer value to the FMU state
        self.output_size = len(self.fmu_model.get_outputs())

    def build(self, input_shape):
        self.input_size = input_shape[-1]
        self.built = True

    def call(self, input_tensor, state):
        output = self.fmu_op(input_tensor, state)
        fmu_state = tf.py_function(
            func=lambda: tf.constant(
                self.fmu_model.get_FMU_state().value,
                dtype=tf.keras.backend.floatx(),
            ),
            inp=[],
            Tout=tf.keras.backend.floatx(),
        )
        return output, [tf.reshape(fmu_state, [1, self.state_size])]

    @tf.custom_gradient
    def fmu_op(self, inputs, state):

        def fmu_step(inputs):
            self.fmu_model.set_inputs(*tf.unstack(tf.reshape(inputs, [-1])))
            self.fmu_model.do_step(self.time, self.dt)
            self.time += self.dt
            return tf.convert_to_tensor(
                self.fmu_model.get_outputs(), dtype=tf.keras.backend.floatx()
            )

        outputs = tf.py_function(fmu_step, inp=[inputs], Tout=tf.keras.backend.floatx())

        def custom_grad(upstream):

            def grad_step(upstream, state, inputs):
                # set the FMU state
                self.fmu_model.set_FMU_state(ctypes.c_void_p(int(state[0])))
                if self.do_step_in_gradient:
                    # do step to compute the directional derivative at the effective current state
                    self.fmu_model.set_inputs(*tf.unstack(tf.reshape(inputs, [-1])))
                    self.fmu_model.do_step(self.time, self.dt)
                return upstream * tf.convert_to_tensor(
                    self.fmu_model.get_directional_derivative(),
                    dtype=tf.keras.backend.floatx(),
                )

            grad = tf.py_function(
                grad_step, inp=[upstream, state, inputs], Tout=tf.keras.backend.floatx()
            )
            return tf.reshape(grad, [1, self.input_size]), None

        return (
            tf.reshape(outputs, [1, self.output_size]),
            custom_grad,
        )

    def reset_states(self):
        # reset time
        self.time = self.start_time
        # reset FMU states
        self.fmu_model.reset_FMU()

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "state_size": self.state_size,
            }
        )
        return config


#  _____ __  __ _   _ _
# |  ___|  \/  | | | | |    __ _ _   _  ___ _ __
# | |_  | |\/| | | | | |   / _` | | | |/ _ \ '__|
# |  _| | |  | | |_| | |__| (_| | |_| |  __/ |
# |_|   |_|  |_|\___/|_____\__,_|\__, |\___|_|
#                                |___/

@tf.keras.utils.register_keras_serializable(package="Custom", name="FMULayer")
class FMULayer(tf.keras.layers.Layer):

    def __init__(
        self,
        fmu_path,
        start_time,
        start_values,
        parameters,
        step_size,
        do_step_in_gradient=False,
        **kwargs
    ):
        super(FMULayer, self).__init__()

        # check if the backend is float64 (needed for FMU state pointer)
        if tf.keras.backend.floatx() != "float64":
            raise ValueError("The FMU layer only supports float64")

        self.units = 1
        self.cell = FMUCell(
            fmu_path,
            start_time,
            start_values,
            parameters,
            step_size,
            do_step_in_gradient=do_step_in_gradient,
        )

        # set initial states
        self.initial_state = tf.constant(
            [[self.cell.fmu_model.get_FMU_state().value]],
            dtype=tf.keras.backend.floatx(),
        )

        # Create the RNN layer with the custom cell
        self.rnn_layer = tf.keras.layers.RNN(self.cell, **kwargs)

    def build(self, input_shape):
        self.cell.build(input_shape)
        self.built = True

    def call(self, inputs):
        self.cell.reset_states()
        return self.rnn_layer(inputs, self.initial_state)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.cell.fmu_model.get_outputs()))
