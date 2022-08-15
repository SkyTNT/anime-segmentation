import argparse
import onnx
import torch
from onnxsim import simplify
from train import AnimeSegmentation


def export_onnx(model, img_size, path):
    torch.onnx.export(model,  # model being run
                      torch.randn(1, 3, img_size, img_size),  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["img"],  # the model's input names
                      output_names=["mask"],  # the model's output names
                      verbose=True
                      )
    onnx_model = onnx.load(path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, path)
    print('finished exporting onnx')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--net', type=str, default='isnet_is',
                        choices=["isnet_is", "isnet", "u2net", "u2netl", "modnet"],
                        help='net name')
    parser.add_argument('--ckpt', type=str, default='saved_models/isnet_best.ckpt',
                        help='model checkpoint path')
    parser.add_argument('--out', type=str, default='saved_models/isnet_best.onnx',
                        help='output path')
    parser.add_argument('--to', type=str, default='onnx', choices=["onnx"],
                        help='export to (format)')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='input image size')
    opt = parser.parse_args()
    print(opt)

    model = AnimeSegmentation.load_from_checkpoint(opt.ckpt, net_name=opt.net, strict=False)
    model.eval()
    if opt.to == "onnx":
        export_onnx(model, opt.img_size, opt.out)
