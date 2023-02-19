import tensorflow as tf

from L1_layers import L1Conv2D, L1Dense

class GradientClippingModel(tf.keras.Model):
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs,
    ):
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)
        self.multiplier = {}
        for var in self.trainable_variables:
            if "conv2d" in var.name:
              self.multiplier[ var.name ] = 0.1*np.sqrt(np.product(var.shape))
            elif "dense" in var.name:
              self.multiplier[ var.name ] = 0.1*np.sqrt(np.product(var.shape))
            elif "batch_normalization" in var.name:
              self.multiplier[ var.name ] = 1.0
            else:
              raise ValueError("layer name can't recognize, found {}".format(var.name))

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # adaptive scaling
        gradients = [grad / tf.norm(grad, ord=2) * self.multiplier[ trainable_vars[idx].name ] for idx,grad in enumerate(gradients)]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

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
    custom_model = GradientClippingModel(inputs=lenet_5bn.inputs, outputs=lenet_5bn.outputs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # epoches*step_per_epoch
    epochs = 50
    batch_size = 256
    lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.1, decay_steps=epochs*np.ceil(len(x_train)/batch_size).astype(np.int32))
    custom_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True, weight_decay=5e-4, clipvalue=1.0),
            loss=loss_fn,
            metrics=['accuracy'])

    # training
    custom_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # validation
    custom_model.evaluate(x_test,  y_test, verbose=2)

if __name__ == '__main__':
    main()
