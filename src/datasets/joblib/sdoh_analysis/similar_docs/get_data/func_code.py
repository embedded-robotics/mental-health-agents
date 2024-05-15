# first line: 13
@memory.cache
def get_data(name):
    return load_dataset(name)
