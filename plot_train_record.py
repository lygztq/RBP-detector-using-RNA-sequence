import matplotlib.pyplot as plt
import os
import numpy as np
from data_utils.class_name import CLASS_INDICES, CLASS_NAMES, CLASS_SIZE

def plot_single_class(cls_name, output_format='png'):
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
    plt.savefig('./train_record/%s_train_record.%s' % (cls_name, output_format), format=output_format)
    plt.close()
    return final_acc

def plot_all():
    with open('./val_acc_record', 'w') as rec_table:
        rec_table.write("class_name\t\t\tclass_size\t\t\tfinal_val_acc\n")
        rec_table.write("-----------------------------------------\n")
        for name in CLASS_NAMES:
            print("ploting %s" % name)
            acc = plot_single_class(name)
            rec_table.write(name + '\t\t\t\t' + str(CLASS_SIZE[name]) + '\t\t\t\t' + str(acc) + '\n')

if __name__ == '__main__':
    plot_all()