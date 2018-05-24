import numpy as np
from model.RNA_model import RNA_model
from data_utils.load_data import write_file
from data_utils.RNA_process import process_batch_result
from data_utils.class_name import CLASS_NAMES

batch_size = 256
use_rnn = False

def single_class_test(cls_name):
    model = RNA_model(
        cls_name=cls_name, batch_size=batch_size,
        test_set_path='./data', is_train=False, use_rnn=use_rnn
    )
    data, result = model.test()
    processed_data, processed_result = process_batch_result(data, result)
    file_path = './data/%s/pred_test' % cls_name
    write_file(file_path, processed_data, with_label=True, label=processed_result)


def test():
    for name in CLASS_NAMES:
        try:
            print("Testing class %s" % name)
            single_class_test(name)
        except:
            print("Can not test class %s, may be the test data file is lost or the model is not trained." % name)

