import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#print(tf.logging.set_verbosity(tf.logging.INFO))

#读入数据包
mnist=input_data.read_data_sets('./')

#print('train_images_shape',mnist.train.images.shape)
#print('train_labels_shape',mnist.train.labels.shape)

#print('validation_images_shape',mnist.validation.images.shape)
#print('validation_labels_shape',mnist.validation.labels.shapes)

#print('test_iamges_shape',mnist.test.images.shape)
#print('test_labels_shape',mnist.test.labels.shape)


plt.figure(figsize=(8,8))

'''for idx in range(16):
    plt.subplot(4,4,idx+1)
    plt.axis('off')#不显示坐标轴
    plt.title('[{}]'.format(mnist.train.labels[idx]))
    plt.imshow(mnist.train.images[idx].reshape((28,28)))
plt.show()'''

x=tf.placeholder('float',[None,784])
y=tf.placeholder('int64',[None])
learning_rate=tf.placeholder('float')

def initialize(shape,stddev=0.1):
    return tf.truncated_normal(shape,stddev=0.1)

#1. 隐层中的神经元个数
L1_units_count=100
W_1=tf.Variable(initialize([784,L1_units_count]))
b_1=tf.Variable(initialize([L1_units_count]))
logits_1=tf.matmul(x,W_1)+b_1
#将乘积数据激活函数，激活函数为ReLU
output_1=tf.nn.relu(logits_1)

#2. 神经网络输出节点 ，共10个输出点
L2_units_count=10
W_2=tf.Variable(initialize([L1_units_count,L2_units_count]))
b_2=tf.Variable(initialize([L2_units_count]))
logits_2=tf.matmul(output_1,W_2)

logits=logits_2

#定义loss和用于优化网络的优化器
cross_entropy_loss=tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits
                                                   ,labels=y)
)
optimizer=tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate
).minimize(cross_entropy_loss)

#softmax概率分类
pred=tf.nn.softmax(logits)
correct_pred=tf.equal(tf.argmax(pred,1),y)
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#saver用于保存或恢复训练的模型
batch_size=32
training_step=1000

saver=tf.train.Saver()

#创建Session，将数据填入网络
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #定义验证集合测试集

    validate_data={
        x:mnist.validation.images,
        y:mnist.validation.labels
    }
    test_data={x:mnist.test.images,y:mnist.test.labels}

    for i in range(training_step):
        xs,ys=mnist.train.next_batch(batch_size)
        _,loss=sess.run(
            [optimizer,cross_entropy_loss],
            feed_dict={
                x:xs,
                y:ys,
                learning_rate:0.3
            }
        )

        #每100次训练打印一次损失值与验证准确率
        if i>0 and i%100==0:
            validate_accuracy=sess.run(accuracy,feed_dict=
                                       validate_data)
            print(
                "after %d training steps,the loss is %g,the validation accuracy is %g"
                %(i,loss,validate_accuracy)
            )
            saver.save(sess,'./model.ckpt',global_step=i)
    print('the training is finish!')
    #最终的测试准确率
    acc=sess.run(accuracy,feed_dict=test_data)
    print('the test accuracy is: ',acc)

with tf.Session() as sess:
    ckpt=tf.train.get_checkpoint_state('./')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        final_pred,acc=sess.run(
            [pred,accuracy],
            feed_dict={
                x:mnist.test.images[:16],
                y:mnist.test.labels[:16]

            }
        )
        orders=np.argsort(final_pred)
        plt.figure(figsize=(8,8))
        print('acc=',acc)
        for idx in range(16):
            order=orders[idx,:][-1]
            prob=final_pred[idx,:][order]
            plt.subplot(4,4,idx+1)
            plt.axis('off')
            plt.title('{}:[{}]-[{:.1f}%]'.format(mnist.test.labels[idx],
                                                 order,prob*100))
            plt.imshow(mnist.test.images[idx].reshape((28,28)))

        plt.show()
    else:
        pass