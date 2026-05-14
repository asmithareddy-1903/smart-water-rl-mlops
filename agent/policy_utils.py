import pickle


def save_policy(q_table, path):

    with open(path, "wb") as f:

        pickle.dump(q_table, f)


def load_policy(path):

    with open(path, "rb") as f:

        return pickle.load(f)