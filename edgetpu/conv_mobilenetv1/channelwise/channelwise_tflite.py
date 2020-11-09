import tensorflow as tf
import numpy as np

import mobilenet_channel

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


def get_flops(set_list, i):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
                model = mobilenet_channel.MobileNet(
                            input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
                            include_top=True, weights=None, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224, 224, 3)), pooling=None,
                            classes=1000, classifier_activation='softmax',
                            ch_64=set_list[i][0], ch_128=set_list[i][1], ch_256=set_list[i][2], ch_512=set_list[i][3], ch_1024=set_list[i][4]
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


b = ["i64", "i128", "i256", "i512", "i1024"]


set_base = [64, 128, 256, 512, 1024, "set_base"]
set1 = [63, 128, 256, 512, 1024, "set1"]
set2 = [65, 128, 256, 512, 1024, "set2"]
set3 = [64, 128, 256, 511, 1024, "set3"]
set4 = [64, 128, 256, 513, 1024, "set4"]

set5 = [64, 96, 256, 512, 1024, "set5"]
set6 = [64, 128, 192, 512, 1024, "set6"]
set7 = [64, 128, 256, 640, 1024, "set7"]
set8 = [64, 128, 256, 512, 768, "set8"]



set_list = [set_base, set1, set2, set3, set4, set5, set6, set7, set8]

FLOPS = []
SUMMARY = []
PARAMS = []

for i in range(len(set_list)):

        model = mobilenet_channel.MobileNet(            
            input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
            include_top=True, weights=None, input_tensor=None, pooling=None,
            classes=1000, classifier_activation='softmax',
            ch_64=set_list[i][0], ch_128=set_list[i][1], ch_256=set_list[i][2], ch_512=set_list[i][3], ch_1024=set_list[i][4]
        )

        model.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
        
        history = model.fit(x_train, y_train, epochs=TRAIN_EPOCHS, 
                            validation_data=(x_test, y_test))
    
        make_tflite(model, "ch_"+set_list[i][5])
        model.summary(print_fn=SUMMARY.append)
        print(get_flops(set_list, i))
        FLOPS.append(get_flops(set_list, i))

for i in range(len(FLOPS)):
    print(FLOPS[i])

for s in SUMMARY:
    if "Total" in s:
        PARAMS.append(s[14:])
print(PARAMS) 
