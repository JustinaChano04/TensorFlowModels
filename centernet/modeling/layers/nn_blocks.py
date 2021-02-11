import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_blocks as official_nn_blocks


class HourglassBlock(tf.keras.layers.Layer):
  """
  Hourglass module
  """

  def __init__(self,
               channel_dims_per_stage,
               blocks_per_stage,
               strides=1,
               **kwargs):
    """
    Args:
      channel_dims_per_stage: list of filter sizes for Residual blocks
      blocks_per_stage: list of residual block repetitions per down/upsample
      strides: integer, stride parameter to the Residual block
    """
    self._order = len(channel_dims_per_stage) - 1
    self._channel_dims_per_stage = channel_dims_per_stage
    self._blocks_per_stage = blocks_per_stage
    self._strides = strides

    assert len(channel_dims_per_stage) == len(blocks_per_stage), 'filter ' \
        'size and residual block repetition lists must have the same length'

    self._filters = channel_dims_per_stage[0]
    self._reps = blocks_per_stage[0]

    super().__init__()

  def build(self, input_shape):
    if self._order == 1:
      # base case, residual block repetitions in most inner part of hourglass
      blocks = [
          official_nn_blocks.ResidualBlock(
              filters=self._filters, strides=self._strides, use_projection=True)
          for _ in range(self._reps)
      ]
      self.blocks = tf.keras.Sequential(blocks)

    else:
      # outer hourglass structures
      main_block = [
          official_nn_blocks.ResidualBlock(
              filters=self._filters, strides=self._strides, use_projection=True)
          for _ in range(self._reps)
      ]
      self.main_block = tf.keras.Sequential(main_block, name='Main_Block')

      side_block = [
          official_nn_blocks.ResidualBlock(
              filters=self._filters, strides=self._strides, use_projection=True)
          for _ in range(self._reps)
      ]
      self.side_block = tf.keras.Sequential(side_block, name='Side_Block')

      self.pool = tf.keras.layers.MaxPool2D(pool_size=2)

      # recursively define inner hourglasses
      self.inner_hg = type(self)(
          channel_dims_per_stage=self._channel_dims_per_stage[1:],
          blocks_per_stage=self._blocks_per_stage[1:],
          strides=self._strides)

      # outer hourglass structures
      end_block = [
          official_nn_blocks.ResidualBlock(
              filters=self._filters, strides=self._strides, use_projection=True)
          for _ in range(self._reps)
      ]
      self.end_block = tf.keras.Sequential(end_block, name='End_Block')

      self.upsample_layer = tf.keras.layers.UpSampling2D(
          size=2, interpolation='nearest')

    super().build(input_shape)

  def call(self, x):
    if self._order == 1:
      return self.blocks(x)
    else:
      x_pre_pooled = self.main_block(x)
      x_side = self.side_block(x_pre_pooled)
      x_pooled = self.pool(x_pre_pooled)
      inner_output = self.inner_hg(x_pooled)
      hg_output = self.end_block(inner_output)
      return self.upsample_layer(hg_output) + x_side
