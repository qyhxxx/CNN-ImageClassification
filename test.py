from skimage import io,transform
import tensorflow as tf
import numpy as np
import os
import xlwt

photo_dict = {0:'1-True',1:'2-Fake',2:'3-Fake'}

w=100
h=100
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h,c))
    return np.asarray(img)


def get_files(filename):
    pic_list = []
    test_files = []
    for pic in os.listdir(filename):
        pic_list.append(pic)
        im = filename + pic
        print('reading the images:%s' % (im))
        img = read_one_image(im)
        test_files.append(img)
    return pic_list,test_files
validation = 'E:/python-projects/dataSets/validation/'
names_data,data = get_files(validation)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('E:/python-projects/model/photo/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('E:/python-projects/model/photo/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())

    wbk = xlwt.Workbook()
    sheet1 = wbk.add_sheet('sheet 1')
    sheet1.write(0, 0, 'picture name')
    sheet1.write(0, 1, 'result')

    sheet2 = wbk.add_sheet('sheet 2')
    sheet2.write(0, 0, 'fake pictures')

    #根据索引通过字典对应图片的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    fake_num = 0
    for i in range(len(output)):
        print(names_data[i]+":"+photo_dict[output[i]])
        sheet1.write(i+1, 0, names_data[i])
        sheet1.write(i+1, 1, photo_dict[output[i]])
        if output[i] != 0:
            sheet2.write(fake_num+1, 0, names_data[i])
            fake_num += 1
    wbk.save('E:/python-projects/dataSets/test.xls')
