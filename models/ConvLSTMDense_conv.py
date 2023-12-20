from imports import *
from preprocessing_params import *
from general_model import GeneralModel, DataAugmentation


build_param_1 = {
    "input_shape": (WINDOW, 1),
    "output_shape": TELESCOPE,
}

compile_param_1 = {
    "loss": tfk.losses.MeanSquaredError(),
    "optimizer": tfk.optimizers.Adam(learning_rate=0.001),
    "metrics": ['mae']
}

fit_param_1 = {
    "batch_size": 256,
    "epochs": 200,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            mode="min",
            min_delta=0.00001,
            restore_best_weights=True
        )
    ],
}


class ConvLSTMDense_conv(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")
        
        x = DataAugmentation(prob=0.3, min_sigma=0.015, max_sigma=0.04)(input_layer)
        
        x = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True, name='lstm'), name='bidirectional_lstm_1')(x)

        x = tfkl.Conv1D(64, 3, padding='same', activation='relu', name='conv_1')(x)

        x = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True, name='lstm'), name='bidirectional_lstm_2')(x)

        x = tfkl.Conv1D(128, 3, padding='same', activation='relu', name='conv_2')(x)

        x = tfkl.Conv1D(256, 3, padding='same', activation='relu', name='conv_3')(x)

        x = tfkl.Flatten()(x)

        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"],
            activation="relu",
            name="Output",
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)
