import numpy as np
import tensorflow as tf

def conv(images, kernel, strides=[1,1], padding='SAME', dilated=1):
    kernel_size = kernel.get_shape()
    images_shape = images.get_shape()
    batch = images_shape[0]
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    kernel_in_ch = kernel_size[2]
    kernel_out_ch = kernel_size[3]
    if padding == 'SAME':
        pad_h, pad_w = (kernel_h-1)//2, (kernel_w-1)//2
    elif padding == 'VALID':
        pad_h, pad_w = (0,0)
    else:
        raise NotImplementedError("padding %s not implemrnt" % padding)

    out_h, out_w = (images_shape[1]-kernel_h+2*pad_h)//strides[0] + 1, (images_shape[2]-kernel_w+2*pad_w)//strides[1] + 1
    
    # im2col
    image_patches = tf.image.extract_patches(images,
                          sizes=[1, kernel_h, kernel_w, 1],
                          strides=[1, strides[0], strides[1], 1],
                          rates=[1, dilated, dilated, 1],   # dilated conv
                          padding=padding)
    image_patches = tf.reshape(image_patches, [batch, out_h, out_w, kernel_h*kernel_w*kernel_in_ch, 1])

    kernel = tf.reshape(kernel, [1, 1, 1, kernel_h*kernel_w*kernel_in_ch, kernel_out_ch])

    mul = tf.multiply(image_patches, kernel)
    result = tf.reduce_sum(mul, 3)
    return result

images = np.random.random((2, 7, 7, 3))
images = tf.convert_to_tensor(images.astype(np.float32))

# [filter_height, filter_width, in_channels, out_channels]
kernel_np = np.random.random((3, 3, 3, 5))
kernel = tf.constant(kernel_np, tf.float32)

pad_valid_actual = conv(images, kernel, strides=[1,1], padding='VALID')
pad_valid_expected = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='VALID')
print("Conv Padding VALID Stride 1 test:")
print(np.allclose(pad_valid_expected, pad_valid_actual))

pad_same_actual = conv(images, kernel, strides=[1,1], padding='SAME')
pad_same_expected = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
print("Conv Padding SAME Stride 1 test:")
print(np.allclose(pad_same_expected, pad_same_actual))

stride2_pad_valid_actual = conv(images, kernel, strides=[2,2], padding='VALID')
stride2_pad_valid_expected = tf.nn.conv2d(images, kernel, strides=[1, 2, 2, 1], padding='VALID')
print("Conv Padding VALID Stride 2 test:")
print(np.allclose(stride2_pad_valid_expected, stride2_pad_valid_actual))

stride2_pad_same_actual = conv(images, kernel, strides=[2,2], padding='SAME')
stride2_pad_same_expected = tf.nn.conv2d(images, kernel, strides=[1, 2, 2, 1], padding='SAME')
print("Conv Padding SAME Stride 2 test:")
print(np.allclose(stride2_pad_same_expected, stride2_pad_same_actual))

stride3_pad_valid_actual = conv(images, kernel, strides=[3,3], padding='VALID')
stride3_pad_valid_expected = tf.nn.conv2d(images, kernel, strides=[1, 3, 3, 1], padding='VALID')
print("Conv Padding VALID Stride 3 test:")
print(np.allclose(stride3_pad_valid_expected, stride3_pad_valid_actual))

stride3_pad_same_actual = conv(images, kernel, strides=[3,3], padding='SAME')
stride3_pad_same_expected = tf.nn.conv2d(images, kernel, strides=[1, 3, 3, 1], padding='SAME')
print("Conv Padding SAME Stride 3 test:")
print(np.allclose(stride3_pad_same_expected, stride3_pad_same_actual))

stride4_pad_valid_actual = conv(images, kernel, strides=[4,4], padding='VALID')
stride4_pad_valid_expected = tf.nn.conv2d(images, kernel, strides=[1, 4, 4, 1], padding='VALID')
print("Conv Padding VALID Stride 4 test:")
print(np.allclose(stride4_pad_valid_expected, stride4_pad_valid_actual))

stride4_pad_same_actual = conv(images, kernel, strides=[4,4], padding='SAME')
stride4_pad_same_expected = tf.nn.conv2d(images, kernel, strides=[1, 4, 4, 1], padding='SAME')
print("Conv Padding SAME Stride 4 test:")
print(np.allclose(stride4_pad_same_expected, stride4_pad_same_actual))
