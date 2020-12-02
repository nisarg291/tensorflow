import tensorflow as tf
print(tf.__version__)
hello = tf.constant("Hello")
world = tf.constant("World")
print(type(hello))
print(hello+world)
# with tf.Session() as sess:
#     result = sess.run(hello+world)
const = tf.constant(10)
mat_1 = tf.fill((5,5),10)
zeros = tf.zeros((5,5))
ones=tf.ones((5,5))
randn=tf.random.normal((5,5),mean=0,stddev=1.0)
randu=tf.random.uniform((5,5),minval=0,maxval=100)
# randn=tf.random_normal_initializer((4,4))
# #randn = tf.random_normal((4,4),mean=0,stddev=1.0)
# randu = tf.random_uniform_initializer((4,4))

print(mat_1)
print(zeros)
print(ones)
print(randn)
print(randu)
a = tf.constant([[1,2],[3,4]])
print(a.get_shape())
b = tf.constant([[10],[100]])
print(b.get_shape())
result=tf.matmul(a,b)
print(result)
#print(tf.Graph())
g=tf.Graph()
print(g is tf.Graph.as_default)
my_tensor=tf.random.uniform((5,5),minval=0,maxval=2)
var1=tf.Variable(my_tensor)
print(var1)










