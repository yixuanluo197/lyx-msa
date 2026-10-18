import json,os,sys
from pathlib import Path
import torch
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
assert torch.cuda.device_count()==1
assert 'TITAN Xp' in torch.cuda.get_device_name(0)
x=torch.arange(12,dtype=torch.float32,device='cuda').reshape(4,3).requires_grad_()
index=torch.tensor([0,0,1,1],device='cuda')
out=scatter_add(x,index,dim=0)
assert torch.isfinite(torch.autograd.grad(out.square().sum(),x)[0]).all()
edge=torch.tensor([[0,1,2,3,0,2],[1,0,3,2,2,0]],device='cuda')
conv=GCNConv(3,4).cuda();opt=torch.optim.Adam(conv.parameters(),lr=.001)
before={n:p.detach().clone() for n,p in conv.named_parameters()}
y=conv(x,edge);y.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in conv.parameters());opt.step()
assert any(not torch.equal(p,before[n]) for n,p in conv.named_parameters())
a=SparseTensor(row=edge[0],col=edge[1],sparse_sizes=(4,4))
z=a.matmul(x);assert torch.isfinite(torch.autograd.grad(z.square().sum(),x)[0]).all()
p={'status':'passed','host_index':os.environ['CMERC_HOST_GPU_INDEX'],'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'logical_device':'cuda:0','name':torch.cuda.get_device_name(0),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_pyg_scatter_sparse_backward':True,'optimizer_parameter_update':True}
Path(sys.argv[1]).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p))
