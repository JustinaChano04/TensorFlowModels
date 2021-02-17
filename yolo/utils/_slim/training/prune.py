import tensorflow as tf
import numpy as np

import modeling.pruning

import keras_flops
import disjoint_set
import more_itertools

def search_channels(cats, i):
    """
    Recursive function that searches all of the convolutions in the model and
    finds out which channels are shared between the convolutions and which ones
    are unique. A dictionary cats will be filled with proxies that will yield
    the filters of a channel. The proxies will have a get_filters function that
    will return a range of filter indices that a convolution corresponds to.
    """
    class Add(list):
        """
        Layer proxy for the addition operation (shortcut in DarkNet)
        """
        def __str__(self):
            return f"Add({super().__str__()})"
        def __repr__(self):
            return f"Add({super().__repr__()})"
        @property
        def filters(self):
            return self[0].filters
        def get_filters(self, conv):
            for i in self:
                if isinstance(i, list):
                    a = i.get_filters(conv)
                    if a:
                        return a
                elif i is conv:
                    return range(i.filters)
            return []

    class Concat(list):
        """
        Layer proxy for the concatenation operation (route in DarkNet)
        """
        def __str__(self):
            return f"Concat({super().__str__()})"
        def __repr__(self):
            return f"Concat({super().__repr__()})"
        @property
        def filters(self):
            return sum(layer.filters for layer in self)
        def get_filters(self, conv):
            count = 0
            for i in self:
                if isinstance(i, list):
                    a = i.get_filters(conv)
                    if a:
                        return a
                elif i is conv:
                    return range(count, i.filters + count)
                count += i.filters
            return []

    if isinstance(i, list) and len(i) == 0:
        return []
    if isinstance(i, tf.keras.layers.Conv2D):
        return i
    elif isinstance(i, tf.keras.layers.Concatenate):
        if i in cats:
            return cats[i]
        else:
            s = Concat()
            for layer in i.inbound_nodes[0].inbound_layers:
                s.append(search_channels(cats, layer))
            cats[i] = s
            return s
    elif isinstance(i, tf.keras.layers.Add):
        s = Add()
        for previous in i.inbound_nodes[0].inbound_layers:
            s.append(search_channels(cats, previous))
        return s
    else:
        previous = i.inbound_nodes[0].inbound_layers
        return search_channels(cats, previous)

def unfold(maybe_list):
    """
    This function iterates over the output of a TensorFlow layer. The output of
    one of these layers.
    """
    if type(maybe_list) is list:
        yield from maybe_list
    elif type(maybe_list) is dict:
        yield from maybe_list.values()
    else:
        yield maybe_list

def prune(input_file, layer_ratio, total_ratio, output_file=None):
    """
    This function is a TensorFlow reimplementation of this function in the
    original paper implementation.
    https://github.com/PengyiZhang/SlimYOLOv3/blob/master/prune.py#L158-L175

    It prunes off excess channels based on which have the lowest impact on
    the final output of the model. This is determined by the weights on the
    batch normalization layers. The lower the weight, the less important
    that channel is to the model and the more safely it can be removed.
    """
    if isinstance(input_file, tf.keras.Model):
        model = input_file
    else:
        model = tf.keras.models.load_model(input_file)
    from modeling.layers import yolo

    # Get all batch norm weights in the model
    weights = []
    for bn in model.submodules:
        if isinstance(bn, tf.keras.layers.BatchNormalization):
            weight_copy = tf.abs(bn.gamma)
            weights.append(weight_copy)
    weights = tf.concat(weights, -1)

    # Find the threshold based on the fraction of channels that we want to prune
    sorted_weights = tf.sort(weights)
    index = int(sorted_weights.shape[0] * total_ratio)
    threshold = sorted_weights[index]

    # Find places in the network in which the number of channels must
    # match up perfectly and store them as keys in an empty dictionary.
    # These areas are only pruned if the channel is redundant in every layer.
    cat_list = []
    dimshare = disjoint_set.DisjointSet()
    for bn in model.submodules:
        # https://github.com/PengyiZhang/SlimYOLOv3/blob/master/prune.py#L36
        if isinstance(bn, tf.keras.layers.Concatenate):
            cat_list.append(bn)
        if not isinstance(bn, tf.keras.layers.Conv2D) and not isinstance(bn, yolo):
            previous = bn.inbound_nodes[0].inbound_layers
            for incoming in unfold(previous):
                dimshare.union(incoming, bn)
    dims = {}
    pending = {}
    for s in dimshare.itersets():
        l = dimshare.find(more_itertools.first(s))
        dims[l] = None

    # Recursively create dictionary to remember the convolutions that the
    # channels for each concatenation come from
    cats = {}
    cats_true = {}
    for i in cat_list:
        search_channels(cats, i)
        cats_true[dimshare.find(i)] = cats[i]
    del cat_list

    # Find all batch normalized convolutions and find any channel that doesn't
    # meet the threshold
    # https://github.com/PengyiZhang/SlimYOLOv3/blob/master/prune.py#L157
    s = modeling.pruning.YOLOSurgeon(model)
    for bn in model.submodules:
        if isinstance(bn, tf.keras.layers.BatchNormalization):
            conv = bn.inbound_nodes[0].inbound_layers

            # gamma in TensorFlow is weight in PyTorch
            # beta in TensorFlow is bias in PyTorch
            weight_copy = tf.abs(bn.gamma)
            channels = weight_copy.shape[0]
            min_channel_num = int(channels * layer_ratio) if int(channels * layer_ratio) > 0 else 1
            mask = (weight_copy > threshold).numpy()

            # prune the unshared channels
            if int(mask.sum()) > min_channel_num:
                channels = np.where(mask)[0]
                root = dimshare.find(conv)
                if root in dims:
                    pending[conv] = channels
                    if dims[root] is None:
                        if root in cats_true:
                            num_channels = cats_true[root].filters
                        dims[root] = set(range(mask.shape[0]))
                    dims[root] -= set(channels)
                else:
                    s.add_job('delete_channels', conv, channels=channels)

    # Prune layers the shared channels
    for conv, channels in pending.items():
        root = dimshare.find(conv)
        shared = dims[root]
        if root in cats_true:
            shared = shared & set(cats_true[root].get_filters(conv))
        s.add_job('delete_channels', conv, channels=list(shared))

    # Save the pruned model
    new_model = s.operate()
    new_model.summary()
    if output_file:
        new_model.save(output_file)
    flops = keras_flops.get_flops(new_model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    return new_model
