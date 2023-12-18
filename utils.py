from imports import *
from preprocessing_params import *

######################### SEQUENCES #########################

def build_sequences(data, valid_periods, window, stride, telescope):
    assert window % stride == 0
    dataset = []
    labels = []
    # iterate over the rows of the dataframe
    for i, time_series in enumerate(data):
        time_series = time_series[valid_periods[i][0] : valid_periods[i][1]]
        time_series = np.expand_dims(time_series, axis=-1)

        padding_check = len(time_series) % window  # division remainder
        if padding_check != 0:
            # Compute padding length
            padding_len = window - len(time_series) % window
            padding = np.zeros((padding_len, 1), dtype="float32")
            time_series = np.concatenate((padding, time_series))
            assert len(time_series) % window == 0

        for idx in np.arange(0, len(time_series) - window - telescope, stride):
            dataset.append(time_series[idx : idx + window])
            labels.append(time_series[idx + window : idx + window + telescope])

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels


def build_multiclass_sequences(
    data, valid_periods, categories, window, stride, telescope
):
    assert window % stride == 0
    dataset_list = []
    labels_list = []
    for cat in np.unique(categories):
        dataset = []
        labels = []
        current_data = data[categories == cat]
        # iterate over the rows of the dataframe
        for i, time_series in enumerate(current_data):
            time_series = time_series[valid_periods[i][0] : valid_periods[i][1]]
            time_series = np.expand_dims(time_series, axis=-1)
            padding_check = len(time_series) % window  # division remainder
            if padding_check != 0:
                # Compute padding length
                padding_len = window - len(time_series) % window
                padding = np.zeros((padding_len, 1), dtype="float32")
                time_series = np.concatenate((padding, time_series))
                assert len(time_series) % window == 0

            for idx in np.arange(0, len(time_series) - window - telescope, stride):
                dataset.append(time_series[idx : idx + window])
                labels.append(time_series[idx + window : idx + window + telescope])

        dataset = np.array(dataset)
        labels = np.array(labels)
        dataset_list.append(dataset)
        labels_list.append(labels)
    return dataset_list, labels_list

######################### PLOTS #########################

def plot_matrix(matrix, save=False, show=True, name="data"):
    # Create a binary mask where 1 represents non-zero values
    non_zero_mask = matrix != 0
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot non-zero values with one color
    ax.imshow(non_zero_mask, cmap="Reds", aspect="auto", interpolation="none")
    ax.axis("off")
    ax.set_title(name)
    # Save the plot
    if save:
        plt.savefig("plot/" + name + ".pdf", format="pdf")
    # Show the plot
    if not show:
        plt.close()


# print n random timeseries of a specific category
def plot_time_series(data, categories=None, category=None, n=5, random=True):
    if category is not None and categories is not None:
        data = data[categories == category]
    # exctract n random indices from 0 to len(data_cat)
    if random:
        indices = np.random.choice(len(data), n, replace=False)
    else:
        indices = np.arange(n)

    figs, axs = plt.subplots(len(indices), 1, sharex=True, figsize=(17, 2 * n))
    for i, idx in enumerate(indices):
        axs[i].plot(data[idx])
        # axs[i].set_title(idx)
    plt.show()


def inspect_multivariate(X, y, telescope, idx=None, n=5):
    if idx == None:
        idx = np.random.randint(0, len(X))

    # Plot three sequences chosen based on idx
    figs, axs = plt.subplots(n, 1, sharex=True, figsize=(30, 3 * n))
    for i in range(idx, idx + n):
        axs[i - idx].plot(np.arange(len(X[i])), X[i])
        axs[i - idx].scatter(
            np.arange(len(X[i]), len(X[i]) + telescope), y[i], color="orange"
        )
        axs[i - idx].set_title("Sequence {}".format(i))
        axs[i - idx].set_ylim(0, 1)
    plt.show()


def inspect_multivariate_prediction(X, y, pred, telescope, idx=None, n=5):
    if idx == None:
        idx = np.random.randint(0, len(X))

    # Plot n sequences chosen based on idx
    figs, axs = plt.subplots(n, 1, sharex=True, figsize=(30, 3 * n))
    for i in range(idx, idx + n):
        axs[i - idx].plot(np.arange(len(X[i])), X[i])
        axs[i - idx].scatter(
            np.arange(len(X[i]), len(X[i]) + telescope), y[i], color="orange"
        )
        axs[i - idx].scatter(
            np.arange(len(X[i]), len(X[i]) + telescope), pred[i], color="green"
        )
        axs[i - idx].set_title("Sequence {}".format(i))
        axs[i - idx].set_ylim(0, 1)
    plt.show()


######################### AUGMENTATIONS #########################

@tf.function
def jitter(x, sigma=0.025):
    # sample the standard deviation
    sigma = tf.random.uniform(shape=[], minval=0.015, maxval=0.03)
    # apply noise to all the time series except the last TELESCOPE time stamps
    x_shape = tf.shape(x)
    noise_shape = (x_shape[0], x_shape[1]-TELESCOPE, x_shape[2])
    zeros_shape = (x_shape[0], TELESCOPE, x_shape[2])
    noise = tf.random.normal(shape=noise_shape, mean=0., stddev=sigma, dtype=tf.float32)
    zeros = tf.zeros(shape=zeros_shape, dtype=tf.float32)
    noise = tf.concat((noise, zeros), axis=1)
    return x + noise