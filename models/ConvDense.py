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
    "epochs": 200,
    "callbacks": [
        tfk.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            mode="min",
            min_delta=0.00001,
            restore_best_weights=True
        ),
        tfk.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.999,
            mode='min',
            min_lr=1e-5
        )
    ],
}


class ConvDense(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):
        tf.random.set_seed(self.seed)

        input_layer = tfkl.Input(shape=self.build_kwargs["input_shape"], name="Input")
        
        #x = DataAugmentation(prob=0.3, min_sigma=0.015, max_sigma=0.02)(input_layer)

        #x = tfkl.Dropout(0.1)(x)
        
        x = tfkl.Conv1D(filters=32, kernel_size=7, activation="relu")(input_layer)
        
        x = tfkl.Conv1D(filters=64, kernel_size=5, activation="relu")(x)
        
        x = tfkl.Conv1D(filters=128, kernel_size=3, activation="relu")(x)

        x = tfkl.Flatten()(x)
        
        x = tfkl.Dense(512, activation="relu")(x)
        x = tfkl.Dense(256, activation="relu")(x)
        
        #x = tfkl.Dropout(0.2)(x)
        
        output_layer = tfkl.Dense(self.build_kwargs["output_shape"], activation="sigmoid")(x)
        # Connect input and output through the Model class
        self.model = tfk.Model(inputs=input_layer, outputs=output_layer, name=self.name)
