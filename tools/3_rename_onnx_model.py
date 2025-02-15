import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'./onnx/best.onnx', help='Path of directory saved the input model.')
    parser.add_argument('--origin_names', default=['x','cast_0.tmp_0'], help='The original name you want to modify.')
    parser.add_argument('--new_names', default=['input','output'], help='The new name you want change to, the number of new_names should be same with the number of origin_names')
    parser.add_argument('--save_file', default=r'./onnx/best_rename.onnx', help='Path to save the new onnx model.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    import onnx
    model = onnx.load(args.model)
    output_tensor_names = set()
    for ipt in model.graph.input:
        output_tensor_names.add(ipt.name)
    for node in model.graph.node:
        for out in node.output:
            output_tensor_names.add(out)

    for origin_name in args.origin_names:
        if origin_name not in output_tensor_names:
            print("[ERROR] Cannot find tensor name '{}' in onnx model graph.".format(origin_name))
            sys.exit(-1)
    if len(set(args.origin_names)) < len(args.origin_names):
        print("[ERROR] There's dumplicate name in --origin_names, which is not allowed.")
        sys.exit(-1)
    if len(args.new_names) != len(args.origin_names):
        print("[ERROR] Number of --new_names must be same with the number of --origin_names.")
        sys.exit(-1)
    if len(set(args.new_names)) < len(args.new_names):
        print("[ERROR] There's dumplicate name in --new_names, which is not allowed.")
        sys.exit(-1)
    for new_name in args.new_names:
        if new_name in output_tensor_names:
            print("[ERROR] The defined new_name '{}' is already exist in the onnx model, which is not allowed.")
            sys.exit(-1)

    for i, ipt in enumerate(model.graph.input):
        if ipt.name in args.origin_names:
            idx = args.origin_names.index(ipt.name)
            model.graph.input[i].name = args.new_names[idx]

    for i, node in enumerate(model.graph.node):
        for j, ipt in enumerate(node.input):
            if ipt in args.origin_names:
                idx = args.origin_names.index(ipt)
                model.graph.node[i].input[j] = args.new_names[idx]
        for j, out in enumerate(node.output):
            if out in args.origin_names:
                idx = args.origin_names.index(out)
                model.graph.node[i].output[j] = args.new_names[idx]

    for i, out in enumerate(model.graph.output):
        if out.name in args.origin_names:
            idx = args.origin_names.index(out.name)
            model.graph.output[i].name = args.new_names[idx]
    
    onnx.checker.check_model(model)
    onnx.save(model, args.save_file)
    print("[Finished] The new model saved in {}.".format(args.save_file))
    print("[DEBUG INFO] The inputs of new model: {}".format([x.name for x in model.graph.input]))
    print("[DEBUG INFO] The outputs of new model: {}".format([x.name for x in model.graph.output]))
