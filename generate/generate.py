import numpy as np
from tensorflow import keras
import random
import copy
from keras.datasets import fashion_mnist
from keras.utils import np_utils
import time

model = keras.models.load_model("model.hdfs")
tapic_img_list=[0,0,0,0,0,0,0,0,0,0]
tapic_tag_list=[0,0,0,0,0,0,0,0,0,0]
methodlist=[0,0,0,0,0]
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
def select_tapic_imgs(test_imgs):
	index=0
	for test_img in test_imgs:
		if index==10:
			break
		actual_res = np.argmax(model.predict(np.expand_dims(test_img, 0)))
		if tapic_tag_list[actual_res]==0:
			tapic_tag_list[actual_res]=1
			tapic_img_list[actual_res]=copy.copy(test_img)
			index+=1
	if index<10:
		for i in range(0,10):
			if tapic_tag_list[i]==0:
				tapic_img_list[i]=test_imgs[0]
def SSIM(img1 ,img2):
    img1
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    C3 = C2/2
    av1 = average(img1)
    av2 = average(img2)
    deviation1 = deviation(img1, av1)
    deviation2 = deviation(img2, av2)
    covariance1 = covariance(img1,av1,img2,av2)
    L = (2*av1*av2 + C1)/(av1*av1 + av2*av2 + C1)
    C = (2 * deviation1 * deviation2 + C2) / (deviation1 * deviation1 + deviation2 * deviation2 + C2)
    S = (covariance1 + C3)/(deviation1*deviation2 + C3)
    return L*C*S
def average(img):
    av = 0.0
    for i in range(28):
        for j in range(28):
            av += img[i][j][0]
    return av/(28*28)
def deviation(img, av):
    deviation = 0.0
    for i in range(28):
        for j in range(28):
            deviation += (img[i][j][0] - av) ** 2
    return (deviation/(28*28 - 1))**0.5
def covariance(img1, av1, img2, av2):
    covariance = 0.0
    for i in range(28):
        for j in range(28):
            covariance += (img1[i][j][0] - av1) * (img2[i][j][0] - av2)
    return covariance/(28*28 - 1)
def diagonal_switch_attack(test_img) :
	actual_res = np.argmax(model.predict(np.expand_dims(test_img, 0)))
	attack_img = copy.copy(test_img)
	for i in range(0,27):
		attack_img[i][i][0]=attack_img[27-i][i][0]
	attack_res = np.argmax(model.predict(np.expand_dims(attack_img, 0)))
	if actual_res!=attack_res:
		return attack_img
	else:
		for i in range(0,27):
			attack_img[15][i][0] = attack_img[i][15][0]
		attack_res = np.argmax(model.predict(np.expand_dims(attack_img, 0)))
		if actual_res != attack_res:
			return attack_img
		else:
			return test_img
def random_switch_attack(test_img):
	time_count = 0
	actual_res = np.argmax(model.predict(np.expand_dims(test_img, 0)))
	while time_count <= 30:
		attack_img = copy.copy(test_img)
		for num in range(0,70):
			random_x = random.randint(0,27)
			random_y = random.randint(0,27)
			random_x2 = random.randint(0,27)
			random_y2 = random.randint(0,27)
			temp = attack_img[random_x][random_y][0]
			attack_img[random_x][random_y][0] =attack_img[random_x2][random_y2]
			attack_img[random_x2][random_y2][0] = temp
		attack_res = np.argmax(model.predict(np.expand_dims(attack_img, 0)))
		if actual_res == attack_res:
			time_count = time_count + 1
		else:
			score = SSIM(test_img, attack_img)
			if score < 0.60:
				continue
			else:
				return attack_img
	return test_img
def tapic_attack(test_img):
	actual_res = np.argmax(model.predict(np.expand_dims(test_img, 0)))
	highest_score=0
	highest_img=copy.copy(test_img)
	for item in range(0,10):
		score=SSIM(tapic_img_list[item],test_img)
		if score>highest_score and item!=actual_res:
			highest_score=score
			highest_img=copy.copy(tapic_img_list[item])
	return highest_img
def attack_img(test_img,ssimsum):
	attack_img = diagonal_switch_attack(test_img)
	actual_res = np.argmax(model.predict(np.expand_dims(test_img, 0)))
	attack_res = np.argmax(model.predict(np.expand_dims(attack_img, 0)))
	if actual_res == attack_res:
		attack_img = random_switch_attack(test_img)
		actual_res = np.argmax(model.predict(np.expand_dims(test_img, 0)))
		attack_res = np.argmax(model.predict(np.expand_dims(attack_img, 0)))
		if actual_res == attack_res:
			attack_img=tapic_attack(test_img)
			actual_res = np.argmax(model.predict(np.expand_dims(test_img, 0)))
			attack_res = np.argmax(model.predict(np.expand_dims(attack_img, 0)))
			if actual_res==attack_res:
				ssimsum += 0
			else:
				ssimsum+=SSIM(test_img,attack_img)
				methodlist[2]+=1
		else:
			ssimsum += SSIM(test_img, attack_img)
			methodlist[1]+=1
	else:
		ssimsum += SSIM(test_img, attack_img)
		methodlist[0]+=1
	return (ssimsum,attack_img)
def generate(images,shape):
	images=images.reshape((images.shape[0],28,28,1))
	generate_images=np.empty_like(images)
	select_tapic_imgs(images)

	ssimsum = 0
	indexofssim = 1
	for test_img in images:
		print("the "+str(indexofssim)+" image is attacking")
		(step,attack_image)=attack_img(test_img,ssimsum)
		step=step-ssimsum
		generate_images[indexofssim-1]=copy.copy(attack_image)
		print("this time ssim is "+str(step))
		ssimsum+=step
		print("the " +str(indexofssim)+" image attack finished, now the ssim is"+str(ssimsum/indexofssim))
		#print("use method A:"+str(methodlist[0])+"  B:"+str(methodlist[1])+"  C:"+str(methodlist[2]))
		indexofssim+=1
	return generate_images.reshape(shape)




# def test2():
# 	aaa=np.load('./attack_data/attack_data.npy')
# 	bbb=np.load('./test_data/test_data.npy')
# 	ssimsum=0
# 	for i in range(0,10000):
# 		ssimsum+=SSIM(aaa[i],bbb[i])
#
# 	print(ssimsum/10000)
#
# test2()

