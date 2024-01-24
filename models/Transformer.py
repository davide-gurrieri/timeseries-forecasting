from imports import *
from preprocessing_params import *
from general_model import GeneralModel, DataAugmentation


build_param_1 = {
    "input_shape": (WINDOW, 1),
    "output_shape": TELESCOPE,
    "head_size": 128,
    "num_heads": 2,
    "ff_dim": 4,
    "num_transformer_blocks": 4,
    "mlp_units": [32],
    "mlp_dropout": 0.3,
    "dropout": 0.15,
}

compile_param_1 = {
    "loss": tfk.losses.MeanSquaredError(),
    "optimizer": tfk.optimizers.Adam(learning_rate=0.001),
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
            restore_best_weights=True,
        )
    ],
}


class Transformer(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = tfkl.LayerNormalization(epsilon=1e-4)(inputs)
        x = tfkl.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = tfkl.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tfkl.LayerNormalization(epsilon=1e-4)(res)
        x = tfkl.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tfkl.Dropout(dropout)(x)
        x = tfkl.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build(self):
        tf.random.set_seed(self.seed)

        inputs = tfk.Input(shape=self.build_kwargs["input_shape"])

        # x = inputs
        x = DataAugmentation(prob=0.4, min_sigma=0.015, max_sigma=0.04)(inputs)

        for _ in range(self.build_kwargs["num_transformer_blocks"]):
            x = self.transformer_encoder(
                x,
                self.build_kwargs["head_size"],
                self.build_kwargs["num_heads"],
                self.build_kwargs["ff_dim"],
                self.build_kwargs["dropout"],
            )

        x = tfkl.GlobalAveragePooling1D(data_format="channels_first")(x)

        for dim in self.build_kwargs["mlp_units"]:
            x = tfkl.Dense(dim, activation="relu")(x)
            x = tfkl.Dropout(self.build_kwargs["mlp_dropout"])(x)

        outputs = tfkl.Dense(
            units=self.build_kwargs["output_shape"], activation="relu", name="Output"
        )(x)

        self.model = tfk.Model(inputs, outputs, name=self.name)
