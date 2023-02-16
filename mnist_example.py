import tensorflow as tf

from custom_conv import L1Conv2D, L1Dense

def main():
    # dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    mean, std = np.mean(x_train), np.std(x_train)
    x_train, x_test = (x_train-mean)/std, (x_test-mean)/std

    lenet_5bn = tf.keras.models.Sequential([
          tf.keras.layers.Input(shape=(28, 28)),
          tf.keras.layers.Reshape((28, 28, 1)),
          tf.keras.layers.Resizing(32, 32),
          # 28*28
          L1Conv2D(kernel_size=[5,5,1,6], strides=[1,1], padding='VALID'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Activation('tanh'),
          # 14*14
          tf.keras.layers.MaxPooling2D((2,2)),
          # 10*10
          L1Conv2D(kernel_size=[5,5,6,16], strides=[1,1], padding='VALID'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Activation('tanh'),
          # 5*5
          tf.keras.layers.MaxPooling2D((2,2)),
          # 1*1
          L1Conv2D(kernel_size=[5,5,16,120], strides=[1,1], padding='VALID'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Activation('tanh'),
          tf.keras.layers.Flatten(),
          L1Dense(84),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Activation('tanh'),
          L1Dense(10)
    ])
    lenet_5bn.summary()

    # build model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # epoches*step_per_epoch
    epochs = 20
    batch_size = 256
    lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.1, decay_steps=epochs*np.ceil(len(x_train)/batch_size).astype(np.int32))
    lenet_5bn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0),
            loss=loss_fn,
            metrics=['accuracy'])

    # training
    lenet_5bn.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # validation
    lenet_5bn.evaluate(x_test,  y_test, verbose=2)

if __name__ == '__main__':
    main()
