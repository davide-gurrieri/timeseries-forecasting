from imports import *
from preprocessing_params import *
import utils


class GeneralModel:
    name = ""
    seed = SEED
    model = tfk.Model()
    history = {}
    history_val = {}
    cv_n_fold = 0
    cv_histories = []
    cv_scores = []
    cv_best_epochs = []
    cv_avg_epochs = -1
    augmentation = None
    base_model = None

    def __init__(self, build_kwargs={}, compile_kwargs={}, fit_kwargs={}):
        self.build_kwargs = build_kwargs
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

    def build(self):
        pass

    def compile(self):
        """
        Compile the model
        """
        self.model.compile(**self.compile_kwargs)

    def train(
        self,
        x_train=None,
        y_train=None,
    ):
        """
        Train the model
        """
        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            **self.fit_kwargs,
        ).history

    def train_val(
        self,
        x_train=None,
        y_train=None,
        x_val=None,
        y_val=None,
    ):
        """
        Train the model
        """
        validation_data = x_val if y_val is None else (x_val, y_val)
        self.history_val = self.model.fit(
            x=x_train,
            y=y_train,
            validation_data=validation_data,
            **self.fit_kwargs,
        ).history

    def save_model(self):
        """
        Save the trained model in the models folder
        """
        self.model.save(f"saved_models/{self.name}")

    def plot_history(self, training=True, figsize=(15, 2)):
        """
        Plot the loss and metrics for the training and validation sets with respect to the training epochs.

        Parameters
        ----------
        training : bool, optional
            show the training plots, by default True
        figsize : tuple, optional
            dimension of the plots, by default (15, 2)
        """
        keys = list(self.history_val.keys())
        n_metrics = len(keys) // 2

        for i in range(n_metrics):
            plt.figure(figsize=figsize)
            if training:
                plt.plot(
                    self.history_val[keys[i]], label="Training " + keys[i], alpha=0.8
                )
            plt.plot(
                self.history_val[keys[i + n_metrics]],
                label="Validation " + keys[i],
                alpha=0.8,
            )
            plt.title(keys[i])
            plt.legend()
            plt.grid(alpha=0.3)

        plt.show()
        
    def predict(self, x_eval):
        predictions = self.model.predict(x_eval, verbose=0)
        return predictions
        
    def evaluate(self, x_eval, y_eval):
        """
        Evaluate the model on the evaluation set.

        Parameters
        ----------
        X_eval : numpy.ndarray
            Evaluation input data
        y_eval : numpy.ndarray
            Evaluation target data
        """
        predictions = self.predict(x_eval)
        mean_squared_error = tfk.metrics.mean_squared_error(y_eval.flatten(), predictions.flatten()).numpy()
        print(f"Mean Squared Error: {mean_squared_error}")
    
    def print_base_model(self):
        for i, layer in enumerate(self.model.get_layer(self.base_model.name).layers):
            print(i, layer.name, layer.trainable)
        
    def unfreeze_layers(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.model.get_layer(self.base_model.name).layers)
        self.model.get_layer(self.base_model.name).trainable = True
        for layer in self.model.get_layer(self.base_model.name).layers[start:end]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable=True
        self.compile()
        

# augmentation
class DataAugmentation(tfk.layers.Layer):
    def __init__(self, prob=0.3, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        
    def call(self, inputs, training=None):
        if training:
            # extract a random number from a uniform distribution between 0 and 1
            random_number = tf.random.uniform(shape=[], minval=0, maxval=1)
            # if the random number is less than the probability, apply the augmentation
            if random_number < self.prob:
                inputs = utils.jitter(inputs)
        return inputs
    

