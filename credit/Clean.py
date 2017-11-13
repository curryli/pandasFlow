# coding=utf-8
import csv

# 读取文件，读取以行为单位，每一行是列表里的一个元素
def read_csv_file(filename, first=1):
    csv_file = csv.reader(open(filename, 'r'))
    return [row for row in csv_file][first:]



if __name__ == '__main__':
    train = read_csv_file('test_certid_date_encrypt.csv')
    result = read_csv_file('pred_label_right.csv')
    fo = open('InnoDeep_1113.csv', 'w')

    index = 0
    for row in result:
        result_id = row[0]
        result_label = row[-1]

        while train[index][0] != result_id:
            fo.write(train[index][0] + ',1\n')
            index += 1

        fo.write(result_id + ',' + result_label + '\n')
        index += 1

    fo.close()
