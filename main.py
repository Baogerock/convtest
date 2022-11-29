from mindspore import Tensor
from typing import Optional

import utlis
from utlis import *

np.random.seed(1024)
input = np.random.randn(1,3,224,224).astype(np.float32)
epoch = 1
def conv1x1(cin, cout, stride=1, bias=False):
  return torch.nn.Conv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias)

x = torch.tensor(input)
pconv1 = conv1x1(3, 3)
torch_params_dict = torch.load("ptest.pth")
pconv1.load_state_dict(torch_params_dict, strict=True)

pre_pout = pconv1(x)
utlis.pytrain(epoch,pconv1,x)
pout = pconv1(x)


ms.set_seed(1024)
class Bottleneck(ms.nn.Cell):

    def __init__(self,
                 stride: int = 1,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[ms.nn.Cell] = None,
                 down_sample: Optional[ms.nn.Cell] = None
                 ) -> None:
        super().__init__()

        self.conv1 = ms.nn.Conv2d(3, 3, kernel_size=1, stride=1,pad_mode='valid')

    def construct(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        return out

x = ms.Tensor(input)
mconv1 = Bottleneck(x)
ckpt_file_name = "./p2m.ckpt"
param_dict = ms.load_checkpoint(ckpt_file_name)
ms.load_param_into_net(mconv1,param_dict)

pre_mout = mconv1(x)
utlis.mstrain(epoch,mconv1,x)
mout = mconv1(x)

print('直接进行卷积:')
print('差值最小:'+str(np.min(np.abs(pre_pout.detach().numpy()-pre_mout.asnumpy()))))
print('差值最大:'+str(np.max(np.abs(pre_pout.detach().numpy()-pre_mout.asnumpy()))))
print('梯度下降'+str(epoch)+'次')
print('差值最小:'+str(np.min(np.abs(pout.detach().numpy()-mout.asnumpy()))))
print('差值最大:'+str(np.max(np.abs(pout.detach().numpy()-mout.asnumpy()))))