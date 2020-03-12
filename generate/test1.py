import generate
from keras.datasets import fashion_mnist
from keras.utils import np_utils
def load_data():
	print("loading images...")
	((train_images, train_labels), (test_images, test_labels)) = fashion_mnist.load_data()
	train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
	test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
	train_images = train_images.astype("float32") / 255.0
	test_images = test_images.astype("float32") / 255.0
	train_labels = np_utils.to_categorical(train_labels, 10)
	test_labels = np_utils.to_categorical(test_labels, 10)
	return (train_images, train_labels), (test_images, test_labels)



#本实例用于演示如何调用generate方法
#((train_images, train_labels), (test_images, test_labels)) = load_data()
#shape1=(100,28,28,1)
#res=generate.generate(test_images[0:100],shape1)
#for i in range(0,100):
#	print(generate.SSIM(res[i],test_images[i]))