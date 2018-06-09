A simple python script that saves the graph of any tensorflow model. The script generates meta file, checkpoints, graph, frozen graph, and optimized frozen graph.  
The following params are required in order to properly save the model:  
* model_directory: Directory to store the model
* model_name: Name of your TensorFlow model
* input_array: Array containing the names of your input TensorFlow variables
* output_array: Array containig the names of your output TensorFlow variables
* forzen_name: Name of the frozen pb graph (default = "frozen.pb")
* optimimzed_name: Name of the optimized frozen pb graph (default = "optimized.pb")

For additional information, check the official documentation [here](https://www.tensorflow.org/programmers_guide/saved_model)
