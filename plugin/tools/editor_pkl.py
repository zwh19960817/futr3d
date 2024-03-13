import os
import pickle

def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    input = '/media/zwh/ZWH4T/ZWH/Dataset3d/nuscenes/nuscenes_infos_val.pkl'
    output = '/media/zwh/ZWH4T/ZWH/Dataset3d/nuscenes/nuscenes_infos_val_mini.pkl'
    data = read_pickle(input)
    data['infos'] = data['infos'][:4]
    write_pickle(data, output)
    pass

