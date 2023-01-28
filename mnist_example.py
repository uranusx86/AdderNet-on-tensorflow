import tensorflow as tf

from custom_conv import L1Conv2D

def main():
    # dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(28, 28)),
      tf.keras.layers.Reshape((28,28,1)),
      # 12*12
      L1Conv2D(kernel_size=[5,5,1,6], strides=[2,2], padding='VALID'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      # 5*5
      L1Conv2D(kernel_size=[3,3,6,8], strides=[2,2], padding='VALID'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      # 2*2
      L1Conv2D(kernel_size=[3,3,8,10], strides=[2,2], padding='VALID'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dropout(0.2),
      # 1*1
      tf.keras.layers.AveragePooling2D((2,2)),
      tf.keras.layers.Flatten()
    ])

    # build model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

    # training
    model.fit(x_train, y_train, epochs=20)

    # validation
    model.evaluate(x_test,  y_test, verbose=2)

if __name__ == '__main__':
    main()
