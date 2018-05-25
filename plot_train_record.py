import matplotlib.pyplot as plt
import os
import numpy as np
from data_utils.class_name import CLASS_INDICES, CLASS_NAMES, CLASS_SIZE

def plot_single_class(cls_name):
    record_path = os.path.join('./train_record', '%s_train_record'%cls_name)
    if not os.path.exists(record_path):
        print('Can not find record file %s' % record_path)
        return
    contant = open(record_path, 'r')
    lines = contant.readlines()
    train_record = map(float, lines[1].split())
    val_record = map(float, lines[3].split())
    final_acc = max(val_record)

    plt.figure(CLASS_INDICES[cls_name])
    plt.grid(True)
    plt.xlabel('number of epoch')
    plt.ylabel('accuracy')
    plt.title('ClassName: %s, DatasetSize: %d, final val accuracy: %f'%(cls_name, CLASS_SIZE[cls_name], final_acc))
    plt.plot(np.array(train_record), 'b-', label="train_acc")
    plt.plot(np.array(val_record), 'y-', label="val_acc")
    plt.legend()
    plt.savefig('./train_record/%s_train_record.png' % cls_name, format='png')
    plt.close()

def plot_all():
    for name in CLASS_NAMES:
        print("ploting %s" % name)
        plot_single_class(name)

if __name__ == '__main__':
    plot_all()