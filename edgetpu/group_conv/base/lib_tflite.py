from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet/')
layers = None

def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000,
              classifier_activation='softmax',
               **kwargs): 
  global layers
  if 'layers' in kwargs:
    layers = kwargs.pop('layers')
  else:
    layers = VersionAwareLayers()
  if kwargs:
    raise ValueError('Unknown argument(s): %s' % (kwargs,))
  if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                     'as true, `classes` should be 1000')

  # Determine proper input shape and default size.
  if input_shape is None:
    default_size = 224
  else:
    if backend.image_data_format() == 'channels_first':
      rows = input_shape[1]
      cols = input_shape[2]
    else:
      rows = input_shape[0]
      cols = input_shape[1]

    if rows == cols and rows in [128, 160, 192, 224]:
      default_size = rows
    else:
      default_size = 224

  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=default_size,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if backend.image_data_format() == 'channels_last':
    row_axis, col_axis = (0, 1)
  else:
    row_axis, col_axis = (1, 2)
  rows = input_shape[row_axis]
  cols = input_shape[col_axis]

  if weights == 'imagenet':
    if depth_multiplier != 1:
      raise ValueError('If imagenet weights are being loaded, '
                       'depth multiplier must be 1')

    if alpha not in [0.25, 0.50, 0.75, 1.0]:
      raise ValueError('If imagenet weights are being loaded, '
                       'alpha can be one of'
                       '`0.25`, `0.50`, `0.75` or `1.0` only.')

    if rows != cols or rows not in [128, 160, 192, 224]:
      rows = 224
      logging.warning('`input_shape` is undefined or non-square, '
                      'or `rows` is not in [128, 160, 192, 224]. '
                      'Weights for input shape (224, 224) will be'
                      ' loaded as the default.')

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  x = _conv_block(img_input, 32, strides=(2, 2), block_id=0)
  
  x = _conv_block(x, 64, block_id=1)

  x = _conv_block(
      x, 128, strides=(2, 2), block_id=2)

  x = _conv_block(x, 128, block_id=3)


  x = _conv_block(
      x, 256, strides=(2, 2), block_id=4)
  
  x = _conv_block(x, 256, block_id=5)

  x = _conv_block(
      x, 512, strides=(2, 2), block_id=6)
  x = _conv_block(x, 512, block_id=7)
  x = _conv_block(x, 512, block_id=8)
  x = _conv_block(x, 512, block_id=9)
  x = _conv_block(x, 512, block_id=10)
  
  x = _conv_block(x, 512, block_id=11)

  x = _conv_block(
      x, 1024, strides=(2, 2), block_id=12)
  
  x = _conv_block(x, 1024, block_id=13)

  if include_top:
    if backend.image_data_format() == 'channels_first':
      shape = (int(1024 * alpha), 1, 1)
    else:
      shape = (1, 1, int(1024 * alpha))

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape(shape, name='reshape_1')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    x = layers.Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = layers.Reshape((classes,), name='reshape_2')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Activation(activation=classifier_activation,
                          name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = training.Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

  # Load weights.
  if weights == 'imagenet':
    if alpha == 1.0:
      alpha_text = '1_0'
    elif alpha == 0.75:
      alpha_text = '7_5'
    elif alpha == 0.50:
      alpha_text = '5_0'
    else:
      alpha_text = '2_5'

    if include_top:
      model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
      weight_path = BASE_WEIGHT_PATH + model_name
      weights_path = data_utils.get_file(
          model_name, weight_path, cache_subdir='models')
    else:
      model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
      weight_path = BASE_WEIGHT_PATH + model_name
      weights_path = data_utils.get_file(
          model_name, weight_path, cache_subdir='models')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model

def _conv_block(inputs, filters, strides=(1, 1), block_id=1):
  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
  #filters = int(filters * alpha)
  #x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad_%d' % block_id)(inputs)
  x = layers.Conv2D(
      filters,
      kernel_size=(3,3),
      padding='same',
      use_bias=False,
      strides=strides,
      name='conv1_%d' % block_id)(
          inputs)
  x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn_%d' % block_id)(x)
  return layers.ReLU(6., name='conv1_relu_%d' % block_id)(x)
