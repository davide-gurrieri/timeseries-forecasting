from imports import *
from preprocessing_params import *
from general_model import GeneralModel


build_param_1 = {
    "input_shape": (WINDOW, 1),
    "output_shape": TELESCOPE,
}

compile_param_1 = {
    "loss": tfk.losses.MeanSquaredError(),
    "optimizer": tfk.optimizers.Adam(learning_rate=0.001),
}

fit_param_1 = {
    "batch_size": 128,
    "epochs": 200,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            mode="min",
            restore_best_weights=True
        ),
        tfk.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            patience=5,
            factor=0.1,
            min_lr=1e-5
        )
    ],
}


class ConvLSTMDenseAttentionConcat(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)

        relu_init = tfk.initializers.HeUniform(seed=self.seed)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        x = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True, name='lstm'), name='bidirectional_lstm')(input_layer)
        
        # Add an attention mechanism
        attention = tfkl.Attention(use_scale=True, score_mode='concat')([x, x])

        # Concatenate the attention output with the LSTM output
        attended_output = tfkl.Concatenate(axis=-1, name='concatenate_attention')([x, attention])

        x = tfkl.Conv1D(32, 3, padding='same', activation='relu', kernel_initializer=relu_init,
         name='conv')(attended_output)
        
        x = tfkl.Flatten()(x)

        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"],
            activation="relu",
            kernel_initializer=relu_init,
            name="Output",
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)