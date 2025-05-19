import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)  # 用于将Python对象序列化并保存到文件中


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
