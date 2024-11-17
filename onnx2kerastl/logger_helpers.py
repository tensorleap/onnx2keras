def log_oom_layers(model, logger):
    # Load the ONNX model
    graph = model.graph
    
    # Log general model information
    logger.debug("IR_VERSION: {}".format(model.ir_version))
    logger.debug("PRODUCER: {}".format(model.producer_name))
    logger.debug("PRODUCER_VERSION: {}".format(model.producer_version))
    logger.debug("DOMAIN: {}".format(model.domain))
    logger.debug("MODEL_VERSION: {}".format(model.model_version))
    logger.debug("OPSET_IMPORT: {}".format(str({opset.domain: opset.version for opset in model.opset_import})))
    
    # Log input information
    for input_tensor in graph.input:
        logger.debug("Input: {}; Shape: {}; Type: {}".format(
            input_tensor.name, 
            str(input_tensor.type.tensor_type.shape.dim).replace('\n', ''),
            input_tensor.type.tensor_type.elem_type
        ))

    # Log output information
    for output_tensor in graph.output:
        logger.debug("Output: {}; Shape: {}; Type: {}".format(
            output_tensor.name,
            str(output_tensor.type.tensor_type.shape.dim).replace('\n', ''), 
            output_tensor.type.tensor_type.elem_type
        ))

    # Log initializers
    for initializer in graph.initializer:
        raw_data_log = initializer.raw_data if len(initializer.raw_data)<50 else ""
        logger.debug("Initializer: {}; Shape: {}; Type: {}; Raw Data: {}".format(
            initializer.name, 
            initializer.dims,
            initializer.data_type,
            raw_data_log
        ))

    # Log each node's information
    for node in graph.node:
        logger.debug("Node: {}; Name: {}".format(node.op_type, node.name))
        logger.debug("  Inputs: {}".format(node.input))
        logger.debug("  Outputs: {}".format(node.output))
        logger.debug("  Attributes:")
        for attr in node.attribute:
            logger.debug("    - {}: {}".format(attr.name, attr))
