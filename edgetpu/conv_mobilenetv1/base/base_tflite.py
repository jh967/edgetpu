import tensorflow as tf
import numpy as np

import model_base

TRAIN_EPOCHS = 3
BLOCK_ITERATION = 10
BATCH_SIZE = 1


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

def make_tflite(model, file_name):
    # Convert the model to tflite_quant.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = False
    tflite_model_quant = converter.convert()
    
    # Save the TFLite_quant.
    
    with tf.io.gfile.GFile('./quant/{}_quant.tflite'.format(file_name), 'wb') as f:
    #with tf.io.gfile.GFile('3_i1024_mbv1_quant.tflite', 'wb') as f:
        f.write(tflite_model_quant)


def get_flops():
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
                model = model_base.MobileNet(
                            input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
                            include_top=True, weights=None, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224, 224, 3)), pooling=None,
                            classes=1000, classifier_activation='softmax'
                                       )
 
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

    tf.compat.v1.reset_default_graph()

    return flops.total_float_ops

x_train = np.random.rand(BATCH_SIZE,224,224,3)
y_train = np.random.rand(BATCH_SIZE)
x_test = np.random.rand(BATCH_SIZE,224,224,3)
y_test = np.random.rand(BATCH_SIZE)

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)


# Load the model from the checkpoint file where the ModelCheckpoint callback saved it to.

'''
model = tf.keras.applications.MobileNet(
    input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
    include_top=True, weights=None, input_tensor=None, pooling=None,
    classes=1000, classifier_activation='softmax'
)
'''

model = model_base.MobileNet(            
        input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
        include_top=True, weights=None, input_tensor=None, pooling=None,
        classes=1000, classifier_activation='softmax'
        )

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
        
history = model.fit(x_train, y_train, epochs=TRAIN_EPOCHS, 
                    validation_data=(x_test, y_test))
    
make_tflite(model, "conv_mobilenetv1_base")
model.summary()
print(get_flops())
