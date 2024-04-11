from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical  # 원-핫 인코딩을 위해 추가

# 데이터를 로드합니다.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 모델을 정의합니다.
network = models.Sequential()
network.add(layers.Input(shape=(28 * 28,)))  # Input 레이어 추가
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))  # 최종 출력 레이어 추가

# 모델 컴파일
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 이미지 데이터를 전처리합니다.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 레이블을 원-핫 인코딩으로 변환합니다.
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 모델을 훈련합니다.
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 모델의 성능을 평가합니다.
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
