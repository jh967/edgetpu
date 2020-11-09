import numpy as np
import tflite_runtime.interpreter as tflite
import time

epoch = 200

def tflite_inference(model,batch_size):
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="{}.tflite".format(model),
           experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    #interpreter = tflite.Interpreter(model_path="{}.tflite".format(model))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_shape = [batch_size,224,224,3]
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    interpreter.resize_tensor_input(input_details[0]['index'], (batch_size,224,224,3))
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    #allocate data
    start1 = time.time()
    interpreter.invoke()
    end1 = time.time()
    
    start2 = time.time()
    for i in range(epoch):
        interpreter.invoke()
    end2 = time.time()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    del(output_data)
    print("batch_size :", '%d' % (batch_size),
          "first_time :", '%.3f' % ((end1 - start1)*1000), "ms",
          "infrence :", '%.3f' % ((end2 - start2)*1000/epoch), "ms",
          " time/batch :", '%.3f' % ((end2 - start2)*1000/(batch_size*epoch)), "ms"
         )  

print("conv_mobilenetv1_quant_edgetpu, epoch: ", epoch)

batch_list = [31,32,33,63,64,65,127,128,129] 

for i in range(len(batch_list)) :
    tflite_inference("./base/edgetpu/conv_mobilenetv1_base_quant_edgetpu", batch_size=batch_list[i])


