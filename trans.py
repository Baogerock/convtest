import torch
from mindspore import save_checkpoint, Tensor
from mindspore import load_checkpoint

#dict转换
pth_file_path='ptest.pth'
#mindspore的网络模型，主要用于获取网络中的各结构的名字
ckpt_file_path='mtest.ckpt'

def convert_model(pth_file_path, torch_key_list, mind_key_list):
    torch_params_dict = torch.load(pth_file_path)
    params_dict_list = []
    i = 0
    for key in mind_key_list:
        value = torch_params_dict[torch_key_list[i]]
        # 先转换为numpy格式，再转换为mindspore的Tensor
        value = Tensor(value.numpy())
        params_dict_list.append({"name": key, "data": value})
        i=i+1
    #保存转换后的文件
    save_checkpoint(params_dict_list, "p2m.ckpt")

#获取pth文件的网络参数名称
torch_params_dict = torch.load(pth_file_path)
torch_key_list = []
for k, v in torch_params_dict.items():
    torch_key_list.append(k)

#获取ckpt文件的网络参数名称
mind_params_dict = load_checkpoint(ckpt_file_path)
mind_key_list = []
for k, v in mind_params_dict.items():
    mind_key_list.append(k)


convert_model(pth_file_path,torch_key_list,mind_key_list)
