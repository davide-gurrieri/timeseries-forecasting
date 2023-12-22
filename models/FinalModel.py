from imports import *
from preprocessing_params import *
from general_model import GeneralModel, DataAugmentation


build_param_1 = {
    "input_shape": (WINDOW, 1),
    "output_shape": TELESCOPE,
}

compile_param_1 = {
    "loss": tfk.losses.MeanSquaredError(),
    "optimizer": tfk.optimizers.Adam(learning_rate=0.001)
}

fit_param_1 = {
    "batch_size": 256,
    "epochs": 400,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            min_delta=0.00001,
            restore_best_weights=True
        ),
        tfk.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=7,
            factor=0.999,
            mode='min',
            min_lr=1e-5
        )
    ],
}

fit_param_2 = {
    "batch_size": 256,
    "epochs": 110,
}


class FinalModel(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="input_layer")
        
        x = DataAugmentation(prob=0.2, min_sigma=0.01, max_sigma=0.02)(input_layer)

        x = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True, name='lstm'), name='bidirectional_lstm')(x)
        
        x = tfkl.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu", name="conv1D_1")(x)
        
        x = tfkl.Dropout(0.1, name = "dropout")(x)
        
        x = tfkl.Conv1D(filters=1, kernel_size=3, padding="same", activation="relu", name="conv1D_2")(x)

        x = tfkl.Flatten(name = "flatten")(x)
        
        output_layer = tfkl.Dense(self.build_kwargs["output_shape"], activation="sigmoid", name = "output_layer")(x)
        
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)
