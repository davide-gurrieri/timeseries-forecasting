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
            monitor="val_loss", patience=6, mode="min", restore_best_weights=True
        ),
        tfk.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", patience=5, factor=0.1, min_lr=1e-5
        ),
    ],
}


class ConvLSTMDenseGAP(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)

        # relu_init = tfk.initializers.HeUniform(seed=self.seed)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")

        x = tfkl.Bidirectional(
            tfkl.LSTM(32, return_sequences=True, name="lstm"), name="bidirectional_lstm"
        )(input_layer)

        x = tfkl.Conv1D(32, 3, padding="same", activation="relu", name="conv")(x)

        x = tfkl.GlobalAveragePooling1D()(x)

        output_layer = tfkl.Dense(
            units=self.build_kwargs["output_shape"],
            activation="relu",
            name="Output",
        )(x)

        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)
