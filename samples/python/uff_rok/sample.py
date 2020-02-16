import os
import sys
import ctypes

# ../common.py
sys.path.insert(1,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir
    )
)
import common

import tensorrt as trt
import graphsurgeon as gs

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

MODEL_PATH = os.path.join(
    WORKING_DIR,
    'models/saved_model.pb'
)

CLIP_PLUGIN_LIBRARY = os.path.join(
    WORKING_DIR,
    'build/librokplugin.so'
)

# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Transforms model path to uff path (e.g. /a/b/c/d.pb -> /a/b/c/d.uff)
def model_path_to_uff_path(model_path):
    uff_path = os.path.splitext(model_path)[0] + ".uff"
    return uff_path

# Converts the TensorFlow frozen graphdef to UFF format using the UFF converter
def model_to_uff(model_path):
    # Transform graph using graphsurgeon to map unsupported TensorFlow
    # operations to appropriate TensorRT custom layer plugins
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph.collapse_namespaces()
    # Save resulting graph to UFF file
    output_uff_path = model_path_to_uff_path(model_path)
    uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        [ModelData.OUTPUT_NAME],
        output_filename=output_uff_path,
        text=True
    )
    return output_uff_path

# Builds TensorRT Engine
def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)

        uff_path = model_to_uff(model_path)
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(uff_path, network)

        return builder.build_cuda_engine(network)


def main():
    # Load the shared object file containing the Clip plugin implementation.
    # By doing this, you will also register the Clip plugin with the TensorRT
    # PluginRegistry through use of the macro REGISTER_TENSORRT_PLUGIN present
    # in the plugin implementation. Refer to plugin/clipPlugin.cpp for more details.
    if not os.path.isfile(CLIP_PLUGIN_LIBRARY):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load library ({}).".format(CLIP_PLUGIN_LIBRARY),
            "Please build the Clip sample plugin.",
            "For more information, see the included README.md"
        ))
    ctypes.CDLL(CLIP_PLUGIN_LIBRARY)

    # Load pretrained model
    if not os.path.isfile(MODEL_PATH):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load model file ({}).".format(MODEL_PATH),
            "Please use 'python lenet5.py' to train and save the model.",
            "For more information, see the included README.md"
        ))

    # Build an engine and retrieve the image mean from the model.
    with build_engine(MODEL_PATH) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    trt_rok = gs.create_plugin_node(name="trt_rok", op="RegionOfKeypoints_TRT", region_shape=5)

if __name__ == "__main__":
    # main()
    import numpy as np
    from scipy.ndimage.filters import maximum_filter
    maximum_filter(np.ones([1, 10, 10]), footprint = [[[0,1,0],[1,1,1],[0,1,0]]])