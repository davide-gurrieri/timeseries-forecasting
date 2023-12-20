from imports import *
from preprocessing_params import *
from general_model import GeneralModel, DataAugmentation


build_param_1 = {
    "input_shape": (WINDOW, 1),
    "output_shape": TELESCOPE,
    "n_feature_maps": 64,
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
            restore_best_weights=True
        )
    ],
}


class Resnet(GeneralModel):
    def __init__(self, name, build_kwargs, compile_kwargs, fit_kwargs):
        super().__init__(build_kwargs, compile_kwargs, fit_kwargs)
        self.name = name

    def build(self):

        input_layer = tfkl.Input(self.build_kwargs["input_shape"])
        noise_layer = tfkl.GaussianNoise(0.4)(input_layer)
        dense_layer_a = tfk.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-5), kernel_initializer=tfk.initializers.HeUniform(SEED))(noise_layer)
        dense_layer_b = tfk.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-5), kernel_initializer=tfk.initializers.HeUniform(SEED))(dense_layer_a)
        dense_layer_c = tfk.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-5), kernel_initializer=tfk.initializers.HeUniform(SEED))(dense_layer_b)
        bid_1 = tfk.layers.Bidirectional(tfk.layers.LSTM(256, return_sequences=True))(dense_layer_c)
        bid_2 = tfk.layers.Bidirectional(tfk.layers.LSTM(256, return_sequences=True))(bid_1)
        dropout = tfkl.Dropout(0.4)(bid_2)

        # BLOCK 1

        conv_x = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"], kernel_size=8, padding='same')(dropout)
        conv_x = tfkl.LayerNormalization()(conv_x)
        conv_x = tfkl.Activation('relu')(conv_x)

        conv_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"], kernel_size=5, padding='same')(conv_x)
        conv_y = tfkl.LayerNormalization()(conv_y)
        conv_y = tfkl.Activation('relu')(conv_y)

        conv_z = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"], kernel_size=3, padding='same')(conv_y)
        conv_z = tfkl.LayerNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"], kernel_size=1, padding='same')(input_layer)
        shortcut_y = tfkl.LayerNormalization()(shortcut_y)

        output_block_1 = tfkl.add([shortcut_y, conv_z])
        output_block_1 = tfkl.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = tfkl.LayerNormalization()(conv_x)
        conv_x = tfkl.Activation('relu')(conv_x)

        conv_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = tfkl.LayerNormalization()(conv_y)
        conv_y = tfkl.Activation('relu')(conv_y)

        conv_z = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = tfkl.LayerNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = tfkl.LayerNormalization()(shortcut_y)

        output_block_2 = tfkl.add([shortcut_y, conv_z])
        output_block_2 = tfkl.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = tfkl.LayerNormalization()(conv_x)
        conv_x = tfkl.Activation('relu')(conv_x)

        conv_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = tfkl.LayerNormalization()(conv_y)
        conv_y = tfkl.Activation('relu')(conv_y)

        conv_z = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = tfkl.LayerNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=1, padding='same')(output_block_2)
        shortcut_y = tfkl.LayerNormalization()(shortcut_y)

        output_block_3 = tfkl.add([shortcut_y, conv_z])
        output_block_3 = tfkl.Activation('relu')(output_block_3)

        # BLOCK 4

        conv_x = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=8, padding='same')(output_block_3)
        conv_x = tfkl.LayerNormalization()(conv_x)
        conv_x = tfkl.Activation('relu')(conv_x)

        conv_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = tfkl.LayerNormalization()(conv_y)
        conv_y = tfkl.Activation('relu')(conv_y)

        conv_z = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = tfkl.LayerNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=1, padding='same')(output_block_3)
        shortcut_y = tfkl.LayerNormalization()(shortcut_y)

        output_block_4 = tfkl.add([shortcut_y, conv_z])
        output_block_4 = tfkl.Activation('relu')(output_block_4)

        # BLOCK 5

        conv_x = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=8, padding='same')(output_block_4)
        conv_x = tfkl.LayerNormalization()(conv_x)
        conv_x = tfkl.Activation('relu')(conv_x)

        conv_y = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = tfkl.LayerNormalization()(conv_y)
        conv_y = tfkl.Activation('relu')(conv_y)

        conv_z = tfkl.Conv1D(filters=self.build_kwargs["n_feature_maps"] * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = tfkl.LayerNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = tfkl.LayerNormalization()(output_block_4)

        output_block_4 = tfkl.add([shortcut_y, conv_z])
        output_block_4 = tfkl.Activation('relu')(output_block_4)

        # FINAL

        gap_layer = tfkl.GlobalAveragePooling1D()(output_block_3)

        dense_layer1 = tfk.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-5), kernel_initializer=tfk.initializers.HeUniform(SEED))(gap_layer)
        dropout = tfkl.Dropout(0.4)(dense_layer1)
        dense_layer2 = tfk.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-5), kernel_initializer=tfk.initializers.HeUniform(SEED))(dense_layer1)
        dropout = tfkl.Dropout(0.3)(dense_layer2)
        dense_layer3 = tfk.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-5), kernel_initializer=tfk.initializers.HeUniform(SEED))(dense_layer2)
        dropout = tfkl.Dropout(0.2)(dense_layer3)
        dense_layer4 = tfk.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(1e-5), kernel_initializer=tfk.initializers.HeUniform(SEED))(dense_layer3)
        
        output_layer = tfkl.Dense(self.build_kwargs["output_shape"], activation='relu')(dense_layer4)

        self.model = tfk.Model(inputs=input_layer, outputs=output_layer)

