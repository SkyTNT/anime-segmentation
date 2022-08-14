import cv2
import numpy as np
import onnx
import torch
from onnxsim import simplify
from torch import nn
import onnxruntime as rt

from model import u2net_refactor
from model import U2NET_full2
from model import U2NET_lite2


def _size_map(x, height):
    # {height: size} for Upsample
    size = [x.shape[2], x.shape[3]]  # 导出时shape会自动转为tensor
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [torch.ceil(w / 2).to(torch.int) for w in size]
    return sizes


u2net_refactor._size_map = _size_map


class AnimeSeg(nn.Module):
    def __init__(self, u2net):
        super().__init__()
        self.u2net = u2net

    def forward(self, x):
        return torch.sigmoid(self.u2net(x)[0])


def convert(model_, x, input_names, output_names, dyn, path):
    model_ = model_.eval()
    torch.onnx.export(model_,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=input_names,  # the model's input names
                      output_names=output_names,  # the model's output names
                      dynamic_axes=dyn,
                      verbose=True
                      )
    onnx_model = onnx.load(path)
    model_simp, check = simplify(onnx_model, dynamic_input_shape=dyn is not None, input_shapes={"img": x.shape})
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, path)
    print('finished exporting onnx')


if __name__ == "__main__":
    # net = U2NET_lite2()
    # net.eval()
    # net.load_state_dict(torch.load('saved_models/best.pt'))
    # anime_seg = AnimeSeg(net)
    # convert(anime_seg, torch.randn((1, 3, 640, 640)), ["img"], ["mask"], {"img": {0: "n", 2: "h", 3: "w"}}, "data/net.onnx")

    seg = rt.InferenceSession("data/net.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    img = cv2.imread(r'data/imgs/000000.jpg')
    h, w = img.shape[:-1]
    h, w = (640, int(640 * w / h)) if h < w else (int(640 * h / w), 640)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (w, h)).astype(np.float32) / 255
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :]
    out = seg.run(None, {'img': img})[0][0]
    out = np.transpose(out * 255, (1, 2, 0)).astype(np.uint8)
    # out = cv2.resize(out, (w, h))
    cv2.imshow("a", out)
    cv2.waitKey()
