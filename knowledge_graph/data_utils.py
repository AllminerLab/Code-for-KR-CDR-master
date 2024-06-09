import pickle


def load_test_data(test_data_path: str):
    """
    :param test_data_path: test data path
    :return: test data = {
        user_1_id: [{'product': product_id, 'score': score}, ...],
        user_2_id: [{'product': product_id, 'score': score}, ...],
        ...
    }
    """
    test_data = pickle.load(open(test_data_path, 'rb'))
    return test_data


def load_attention_network_train_data(attention_network_train_data_path: str):
    """
    :param attention_network_train_data_path: attention network train data path
    :return: attention network train data = {
        user_1_id: {
            'positive': [{'product': product_id, 'score': score}, ...],
            'negative': [{'product': product_id, 'score': score}, ...]
        },
        user_2_id: {
            'positive': [{'product': product_id, 'score': score}, ...],
            'negative': [{'product': product_id, 'score': score}, ...]
        },
        ...
    }
    """
    attention_network_train_data = pickle.load(open(attention_network_train_data_path, 'rb'))
    return attention_network_train_data


def load_meta_network_train_data(meta_network_train_data_path: str):
    """
    :param meta_network_train_data_path: meta network train data path
    :return: meta network train data = {
        user_1_id: {
            'positive': [{'product': product_id, 'score': score}, ...],
            'negative': [{'product': product_id, 'score': score}, ...]
        },
        user_2_id: {
            'positive': [{'product': product_id, 'score': score}, ...],
            'negative': [{'product': product_id, 'score': score}, ...]
        },
        ...
    }
    """
    meta_network_train_data = pickle.load(open(meta_network_train_data_path, 'rb'))
    return meta_network_train_data