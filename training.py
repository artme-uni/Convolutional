import pickle
import random
import numpy as np

NUMBER_OF_CLASSES: int = 10
IMAGE_SIZE: int = 28

Params = list[np.ndarray]
Outputs = list[np.ndarray]
Gradients = list[np.ndarray]
Vs = list[np.ndarray]


def shuffle(data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = list(range(len(data)))
    random.shuffle(indices)
    return data[indices], target[indices]


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    from keras.datasets import mnist
    (train_data, train_target), (test_data, test_target) = mnist.load_data()

    def prepare(array: np.ndarray) -> np.ndarray:
        return array.astype(np.float32).reshape((len(array), 1, IMAGE_SIZE, IMAGE_SIZE)) / 255

    return (prepare(train_data), train_target), (prepare(test_data), test_target)


def to_one_hot(cls: int) -> np.ndarray:
    code: np.ndarray = np.zeros(shape=(NUMBER_OF_CLASSES, 1))
    code[cls][0] = 1.0
    return code


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    return -(q * np.log(p)).sum()


def update(
        params: Params,
        v: Vs,
        gradients: Gradients,
        gamma: float,
        lr: float,
        batch_size: int) -> None:
    for i in range(len(params)):
        v[i] = gamma * v[i] + lr * gradients[i] / batch_size
        params[i] -= v[i]


def create_accumulator(params: Params) -> Gradients:
    return [np.zeros_like(params[i]) for i in range(len(params))]


def accumulate_gradients(
        accumulator: Gradients,
        gradients: Gradients) -> None:
    for i in range(len(accumulator)):
        accumulator[i] += gradients[i]


def nesterov_gd(
        batch_data: np.ndarray,
        batch_target: np.ndarray,
        lr: float, gamma: float,
        params: Params,
        vs: Vs) -> (Params, Vs, float):
    cost: float = 0
    batch_size = len(batch_data)

    gradients: list[np.ndarray] = create_accumulator(params)

    for i in range(batch_size):
        data: np.ndarray = batch_data[i]
        target: np.ndarray = to_one_hot(batch_target[i])

        outputs = feed_forward(data, params)
        current_gradients: Gradients = compute_gradients(data, target, outputs, params)

        accumulate_gradients(gradients, current_gradients)

        cost += cross_entropy(outputs[-1], target)

    update(params, vs, gradients, gamma, lr, batch_size)

    cost = cost / batch_size

    return params, vs, cost


def cross_corr_bp(
        errors: np.ndarray,
        image: np.ndarray,
        filters: np.ndarray,
        stride: int) -> tuple[np.ndarray, np.ndarray]:
    filters_count, filter_channels, filter_size, _ = filters.shape
    _, image_size, _ = image.shape

    next_errors: np.ndarray = np.zeros_like(image)
    gradients: np.ndarray = np.zeros_like(filters)

    for filter_index in range(filters_count):
        filter: np.ndarray = filters[filter_index]
        gradient: np.ndarray = gradients[filter_index]
        error: np.ndarray = errors[filter_index]

        y: int = 0
        for offset_y in range(0, image_size - filter_size + 1, stride):
            x: int = 0
            for offset_x in range(0, image_size - filter_size + 1, stride):
                gradient += error[y, x] * image[:, offset_y:offset_y + filter_size, offset_x:offset_x + filter_size]
                next_errors[:, offset_y:offset_y + filter_size, offset_x:offset_x + filter_size] += error[y, x] * filter
                x += 1
            y += 1

    return next_errors, gradients


def argmax2d(arr: np.ndarray) -> tuple[int, int]:
    idx: int = arr.argmax()
    return idx // arr.shape[1], idx % arr.shape[1]


def maxpool_bp(
        errors: np.ndarray,
        image: np.ndarray,
        patch_size: int,
        stride: int) -> np.ndarray:
    channels_count, image_size, _ = image.shape

    next_errors: np.ndarray = np.zeros_like(image)

    for channel_index in range(channels_count):
        channel: np.ndarray = image[channel_index]
        channel_error: np.ndarray = next_errors[channel_index]
        error: np.ndarray = errors[channel_index]

        out_y: int = 0
        for curr_y in range(0, image_size - patch_size + 1, stride):
            out_x: int = 0
            for curr_x in range(0, image_size - patch_size + 1, stride):
                (y, x) = argmax2d(channel[curr_y:curr_y + patch_size, curr_x:curr_x + patch_size])
                channel_error[curr_y + y, curr_x + x] = error[out_y, out_x]
                out_x += 1
            out_y += 1

    return next_errors


def dense_bp(
        error: np.ndarray,
        layer: np.ndarray,
        layer_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad = error.dot(layer_in.T)
    error = layer.T.dot(error)

    return error, grad


def compute_gradients(
        image: np.ndarray,
        target: np.ndarray,
        outputs: Outputs,
        params: Params) -> Gradients:
    conv1, conv2, fc1, fc2 = params
    conv1_out, conv2_out, downsampled, flattened, fc1_out, result = outputs

    error: np.ndarray = result - target

    error, fc2_grad = dense_bp(error, fc2, fc1_out)
    error[fc1_out <= 0] = 0

    error, fc1_grad = dense_bp(error, fc1, flattened)
    error = error.reshape(downsampled.shape)

    error = maxpool_bp(error, conv2_out, patch_size=2, stride=2)
    error[conv2_out <= 0] = 0

    error, conv2_grad = cross_corr_bp(error, conv1_out, conv2, stride=1)
    error[conv1_out <= 0] = 0

    _, conv1_grad = cross_corr_bp(error, image, conv1, stride=1)

    grads = [conv1_grad, conv2_grad, fc1_grad, fc2_grad]

    return grads


def cross_corr(
        image: np.ndarray,
        filters: np.ndarray,
        stride: int) -> np.ndarray:
    filters_count, filter_channels, filter_size, _ = filters.shape
    image_channels, image_size, _ = image.shape

    feature_map_size: int = (image_size - filter_size) // stride + 1
    feature_maps: np.ndarray = np.zeros(shape=(filters_count, feature_map_size, feature_map_size))

    for filter_index in range(filters_count):
        filter: np.ndarray = filters[filter_index]
        feature_map: np.ndarray = feature_maps[filter_index]

        y: int = 0
        for offset_y in range(0, image_size - filter_size + 1, stride):
            x: int = 0
            for offset_x in range(0, image_size - filter_size + 1, stride):
                feature_map[y][x] = \
                    (filter * image[:, offset_y:offset_y + filter_size, offset_x:offset_x + filter_size]).sum()
                x += 1
            y += 1

    return feature_maps


def maxpool(
        image: np.ndarray,
        patch_size: int,
        stride: int):
    image_channels, image_size, _ = image.shape
    downsampled_size: int = (image_size - patch_size) // stride + 1
    downsampled: np.ndarray = np.zeros(shape=(image_channels, downsampled_size, downsampled_size))

    for channel_index in range(image_channels):
        channel: np.ndarray = image[channel_index]
        downsampled_channel: np.ndarray = downsampled[channel_index]

        y: int = 0
        for offset_y in range(0, image_size - patch_size + 1, stride):
            x: int = 0
            for offset_x in range(0, image_size - patch_size + 1, stride):
                downsampled_channel[y, x] = \
                    channel[offset_y:offset_y + patch_size, offset_x:offset_x + patch_size].max()
                x += 1
            y += 1

    return downsampled


def softmax(z: np.ndarray) -> np.ndarray:
    exp: np.ndarray = np.exp(z)
    return exp / exp.sum()


def feed_forward(
        image: np.ndarray,
        params: Params) -> Outputs:
    conv1, conv2, fc1, fc2 = params

    conv1_out: np.ndarray = cross_corr(image, conv1, stride=1)
    conv1_out[conv1_out <= 0] = 0

    conv2_out: np.ndarray = cross_corr(conv1_out, conv2, stride=1)
    conv2_out[conv2_out <= 0] = 0
    downsampled: np.ndarray = maxpool(conv2_out, patch_size=2, stride=2)

    maps_count, map_size, _ = downsampled.shape
    flattened: np.ndarray = downsampled.reshape((maps_count * map_size * map_size, 1))

    fc1_out: np.ndarray = fc1.dot(flattened)
    fc1_out[fc1_out <= 0] = 0

    fc2_out: np.ndarray = fc2.dot(fc1_out)
    result: np.ndarray = softmax(fc2_out)

    return [conv1_out, conv2_out, downsampled, flattened, fc1_out, result]


def classify(image: np.ndarray, params: Params) -> tuple[int, np.ndarray]:
    result: np.ndarray = feed_forward(image, params)[-1]
    return result.argmax(), result


def create_filters(size: tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    stddev: float = scale / np.sqrt(np.array(size).prod())
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weights(size: tuple[int, ...]) -> np.ndarray:
    return np.random.standard_normal(size=size) * 0.01


def train(
        lr: float,
        gamma: float,
        parameters_file_name: str,
        first_layer_size: int,
        second_layer_size: int,
        first_layer_filters_count,
        second_layer_filters_count,
        third_layer_size,
        fourth_layer_size
) -> None:
    (data, target), _ = load_mnist()

    batch_size = 60

    batches: list[tuple[np.ndarray, np.ndarray]] = [
        (data[i:i + batch_size], target[i:i + batch_size])
        for i in range(0, len(data), batch_size)]

    filters1: np.ndarray = create_filters((first_layer_filters_count, 1, first_layer_size, first_layer_size))
    filters2: np.ndarray = create_filters(
        (second_layer_filters_count, first_layer_filters_count, second_layer_size, second_layer_size))
    second_layer_filters_size = (IMAGE_SIZE - first_layer_size + 1 - second_layer_size + 1)
    fc1_input = int(second_layer_filters_size * second_layer_filters_size * second_layer_filters_count / 4)
    fc1: np.ndarray = initialize_weights((third_layer_size, fc1_input))
    fc2: np.ndarray = initialize_weights((fourth_layer_size, third_layer_size))

    params: Params = [filters1, filters2, fc1, fc2]

    cost: list[float] = []

    vs: Vs = [np.zeros_like(filters1), np.zeros_like(filters2), np.zeros_like(fc1), np.zeros_like(fc2)]

    for batch in batches:
        data, target = batch
        params, vs, current_cost = nesterov_gd(data, target, lr, gamma, params, vs)
        cost.append(current_cost)

    with open(parameters_file_name + ".pkl", 'wb') as parameters_file:
        pickle.dump([params, cost], parameters_file)
