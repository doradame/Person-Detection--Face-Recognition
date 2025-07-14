import onnx
import onnxruntime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration to find the model path
try:
    import config
    MODEL_PATH = config.INSIGHTFACE_MODEL_PATH
except (ImportError, AttributeError) as e:
    logging.error(f"Could not load configuration or find INSIGHTFACE_MODEL_PATH: {e}")
    # Use a fallback path if the configuration is not available
    MODEL_PATH = os.environ.get("INSIGHTFACE_MODEL_PATH", "models/insightface_model.onnx")


def inspect_onnx_model(model_path: str):
    """
    Loads an ONNX model and inspects its inputs and outputs.
    """
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: '{model_path}'")
        return

    logging.info(f"Inspecting ONNX model: {model_path}")

    try:
        # Load the model without running it
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        logging.info("The ONNX model is valid.")

        # Print information about model inputs
        inputs = model.graph.input
        logging.info(f"Number of model inputs: {len(inputs)}")
        for i, input_tensor in enumerate(inputs):
            name = input_tensor.name
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            dtype = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
            logging.info(f"  Input #{i+1}:")
            logging.info(f"    Name: {name}")
            logging.info(f"    Data type: {dtype}")
            logging.info(f"    Shape: {shape} (Batch, Channels, Height, Width)")

        # Print information about model outputs
        outputs = model.graph.output
        logging.info(f"\nNumber of model outputs: {len(outputs)}")
        for i, output_tensor in enumerate(outputs):
            name = output_tensor.name
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            dtype = onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)
            logging.info(f"  Output #{i+1}:")
            logging.info(f"    Name: {name}")
            logging.info(f"    Data type: {dtype}")
            logging.info(f"    Shape: {shape}")

    except Exception as e:
        logging.error(f"Error inspecting the ONNX model: {e}", exc_info=True)

    logging.info("\n--- Inspection via ONNX Runtime ---")
    try:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Input
        inputs_rt = session.get_inputs()
        logging.info(f"Number of inputs (runtime): {len(inputs_rt)}")
        for i, input_meta in enumerate(inputs_rt):
            logging.info(f"  Input #{i+1}:")
            logging.info(f"    Name: {input_meta.name}")
            logging.info(f"    Shape: {input_meta.shape}")
            logging.info(f"    Type: {input_meta.type}")

        # Output
        outputs_rt = session.get_outputs()
        logging.info(f"\nNumber of outputs (runtime): {len(outputs_rt)}")
        for i, output_meta in enumerate(outputs_rt):
            logging.info(f"  Output #{i+1}:")
            logging.info(f"    Name: {output_meta.name}")
            logging.info(f"    Shape: {output_meta.shape}")
            logging.info(f"    Type: {output_meta.type}")

    except Exception as e:
        logging.error(f"Error during inspection with ONNX Runtime: {e}", exc_info=True)


if __name__ == "__main__":
    inspect_onnx_model(MODEL_PATH)
