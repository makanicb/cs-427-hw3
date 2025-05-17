import pytest
from lr import \
    (sigmoid_function,
     sigmoid_function_stable,
     train_lr,
     read_data,
     dot,
     vec_scale,
     vec_add,
     vec_sub,
     vec_mag,
     vec_norm,
     compute_gradient_of_weights,
     predict_lr,
     compute_gradient_of_bias,
     get_labels_from_data,
     get_examples_from_data,
     convert_labels)


def test_sigmoid():

    assert sigmoid_function(0) == 0.5
    assert round(sigmoid_function(1), 3) == 0.731
    assert round(sigmoid_function(-1), 3) == 0.269


def test_sigmoid_stable():
    assert sigmoid_function_stable(0) == 0.5
    assert sigmoid_function_stable(99999) > 0.9
    assert sigmoid_function_stable(-99999) < 0.1


def test_train_lr_OR():
    # OR function
    data = [
        ([0, 0], -1),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]

    eta = 0.1
    l2_reg_weight = 0.1

    model = train_lr(data, eta, l2_reg_weight)

    # assert predict_lr(model, data[0][0]) < 0.50
    # assert predict_lr(model, data[1][0]) > 0.50
    # assert predict_lr(model, data[2][0]) > 0.50
    # assert predict_lr(model, data[3][0]) > 0.50

    print("\n", model)
    for x, label in data:
        prob = predict_lr(model, x)
        print(f"x = {x}, predicted = {prob:.4f}, actual = {label}")


def test_train_lr_AND():
    # OR function
    data = [
        ([0, 0], -1),
        ([0, 1], -1),
        ([1, 0], -1),
        ([1, 1], 1)
    ]

    eta = 0.1
    l2_reg_weight = 0.0

    model = train_lr(data, eta, l2_reg_weight)

    # assert predict_lr(model, data[0][0]) < 0.50
    # assert predict_lr(model, data[1][0]) < 0.50
    # assert predict_lr(model, data[2][0]) < 0.50
    # assert predict_lr(model, data[3][0]) > 0.50

    print("\n", model)
    for x, label in data:
        prob = predict_lr(model, x)
        print(f"x = {x}, predicted = {prob:.4f}, actual = {label}")


def test_read_data():
    data, varnames = read_data(r'..\spambase\spambase-train.csv')
    examples = [example for (example, label) in data]
    labels = [label for (example, label) in data]
    print(data)


def test_dot():
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert dot(a, b) == 32


def test_scale_vec():
    vec = [1, 2, 3]
    salar = 2
    assert vec_scale(vec, salar) == [2, 4, 6]


def test_add_vec():
    vec_a = [1, 2, 3]
    vec_b = [4, 5, 6]
    assert vec_add(vec_a, vec_b) == [5, 7, 9]


def test_vec_sub():
    vec_a = [1, 2, 3]
    vec_b = [4, 5, 6]
    assert vec_sub(vec_a, vec_b) == [-3, -3, -3]


def test_vec_mag():
    vec = [3, 3]
    assert round(vec_mag(vec), 3) == 4.243


def test_vec_mag_zero():
    vec = [0, 0]
    assert vec_mag(vec) == 0


def test_vec_norm():
    vec = [4, 0]
    assert vec_norm(vec) == [1, 0]


def test_compute_gradient_of_weights():
    x = [
        [1,2],
        [2,1]
    ]
    w = [0, 0]
    y_hat = [0.5, 0.5]
    y = [1, 0]
    l2_reg_weight = 0.1

    # Assertion fails following modification of gradient descent in which w
    # no longer average the gradient over the number of examples
    assert(compute_gradient_of_weights(x, w, y_hat, y, l2_reg_weight) == [0.25, -0.25])


def test_predict_lr():
    # Simply tests that sigma(dot(x,w)) is 0.5 when dot(x,w) is 0.
    x = [
        [1,0],
        [2,0]
    ]
    w = [0, 1]
    b = 0.0
    model = (w,b)
    assert predict_lr(model, x[0]) == 0.5
    assert predict_lr(model, x[1]) == 0.5


def test_compute_gradient_of_bias():
    y = [1,1,0,0]
    y_hat = [0,1,1,0]
    assert compute_gradient_of_bias(y_hat, y) == 0

    y = [1,1,0,0]
    y_hat = [0,1,1,1]
    assert compute_gradient_of_bias(y_hat, y) == 0.25


def test_get_examples_from_data():
    data = [
        ([0, 0], 1),
        ([1, 1], 0)
    ]
    assert get_examples_from_data(data) == [[0,0], [1,1]]


def test_get_labels_from_data():
    data = [
        ([0, 0], 1),
        ([1, 1], 0)
    ]
    assert get_labels_from_data(data) == [1, 0]


def test_convert_labels():
    labels = [-1, 1, 1, -1]
    assert convert_labels(labels) == [0, 1, 1, 0]

def test_convert_labels_same():
    labels = [0, 1, 1, 0]
    assert convert_labels(labels) == [0, 1, 1, 0]
