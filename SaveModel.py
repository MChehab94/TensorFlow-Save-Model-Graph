import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


def save_model(model_directory, model_name, input_array, output_array, frozen_name='frozen.pb', optimized_name='optimized.pb'):
    input_graph_path = model_directory + '/' + model_name + '.pbtxt'
    checkpoint_path = model_directory + '/' + model_name + '.ckpt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "prediction"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = model_directory + '/' + frozen_name
    output_optimized_graph_name = model_directory + '/' + optimized_name
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_array,
        output_array,
        tf.float32.as_datatype_enum)

    # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())