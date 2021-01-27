from statistics import mean


def k_fold_validation(data, labels, k, model_constructor, *model_args, **model_kwargs):
    fold_size = len(data) // k
    errors = []

    for i in range(k):
        test_data = data[i * fold_size: i * fold_size + fold_size]
        test_labels = labels[i * fold_size: i * fold_size + fold_size]

        training_ids = list(range(0, i * fold_size)) + list(range(i * fold_size + fold_size, len(data)))
        training_data = data[training_ids]
        training_labels = labels[training_ids]

        model = model_constructor(data=training_data, labels=training_labels, *model_args, **model_kwargs)
        errors.append(error_rate(test_data, test_labels, model))

    return mean(errors)


def k_fold_model_and_error(data, labels, k, model_constructor, *model_args, **model_kwargs):
    """
    :return: tuple(model_constructor: model, float: error)
            model: model trained on the entire dataset
            error: model error estimated from k-fold cross validation
    """
    model = model_constructor(data=data, labels=labels, *model_args, **model_kwargs)
    error = k_fold_validation(data, labels, k, model_constructor, *model_args, **model_kwargs)

    return model, error


def error_rate(test_data, test_labels, model):
    wrong_predicts = 0
    for data, label in zip(test_data, test_labels):
        if model.predict(data) != label:
            wrong_predicts += 1
    if len(test_data) == 0:
        return 0
    return wrong_predicts / len(test_data)

# train_correct = 0
# for i in range(START, STOP):
#     if tree.predict(images[i]) == labels[i]:
#         train_correct += 1
