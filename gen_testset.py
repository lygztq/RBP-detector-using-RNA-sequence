from data_utils.class_name import CLASS_NAMES
from data_utils.load_data import read_file, write_file

test_class = CLASS_NAMES[35]

def generate_dev_test_set(cls_name):
    path = './data/%s/train' % cls_name
    data, _ = read_file(path, with_label=True)
    test_set_path = './data/%s/test' % cls_name
    write_file(test_set_path, data)


if __name__ == "__main__":
    generate_dev_test_set(test_class)
    
