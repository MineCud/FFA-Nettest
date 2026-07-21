import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
print('log_dir :',log_dir)
print('model_name:',model_name)

models_={
	'ffa':FFA(gps=opt.gps,blocks=opt.blocks),
}
loaders_={
	'its_train':ITS_train_loader,
	'its_test':ITS_test_loader,
	'ots_train':OTS_train_loader,
	'ots_test':OTS_test_loader,
	'rs_train':RS_train_loader,
	'rs_test':RS_test_loader,
	'rrshid_train':RRSHID_train_loader,
	'rrshid_val':RRSHID_val_loader,
	'rrshid_test':RRSHID_test_loader,
	'rrshid_tn_train':RRSHID_tn_train_loader,
	'rrshid_tn_val':RRSHID_tn_val_loader,
	'rrshid_tn_test':RRSHID_tn_test_loader,
	'rrshid_m_train':RRSHID_m_train_loader,
	'rrshid_m_val':RRSHID_m_val_loader,
	'rrshid_m_test':RRSHID_m_test_loader,
	'rrshid_tk_train':RRSHID_tk_train_loader,
	'rrshid_tk_val':RRSHID_tk_val_loader,
	'rrshid_tk_test':RRSHID_tk_test_loader,
}
start_time=time.time()
T=opt.steps

def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def load_checkpoint(net, ckp_path, tag='checkpoint'):
	ckp=torch.load(ckp_path, map_location=opt.device)
	state=ckp['model'] if isinstance(ckp, dict) and 'model' in ckp else ckp
	net.load_state_dict(state)
	print(f'loaded {tag} from {ckp_path}')
	return ckp if isinstance(ckp, dict) else {}

def train(net,loader_train,loader_test,optim,criterion):
	losses=[]
	start_step=0
	max_ssim=0
	max_psnr=0
	ssims=[]
	psnrs=[]

	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp=load_checkpoint(net, opt.model_dir, tag='resume checkpoint')
		losses=ckp.get('losses', [])
		start_step=ckp.get('step', 0)
		max_ssim=ckp.get('max_ssim', 0)
		max_psnr=ckp.get('max_psnr', 0)
		psnrs=ckp.get('psnrs', [])
		ssims=ckp.get('ssims', [])
		print(f'start_step:{start_step} start training ---')
	elif opt.pretrain and os.path.exists(opt.pretrain):
		load_checkpoint(net, opt.pretrain, tag='pretrain')
		print('fine-tune from pretrain ***')
	else:
		print('train from scratch ***')

	for step in range(start_step+1,opt.steps+1):
		net.train()
		lr=opt.lr
		if not opt.no_lr_sche:
			lr=lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr
		x,y=next(iter(loader_train))
		x=x.to(opt.device);y=y.to(opt.device)
		out=net(x)
		loss=criterion[0](out,y)
		if opt.perloss:
			loss2=criterion[1](out,y)
			loss=loss+0.04*loss2

		loss.backward()
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

		if step % opt.eval_step ==0 :
			with torch.no_grad():
				ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,step)

			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			if ssim_eval > max_ssim and psnr_eval > max_psnr :
				max_ssim=max(max_ssim,ssim_eval)
				max_psnr=max(max_psnr,psnr_eval)
				torch.save({
							'step':step,
							'max_psnr':max_psnr,
							'max_ssim':max_ssim,
							'ssims':ssims,
							'psnrs':psnrs,
							'losses':losses,
							'model':net.state_dict()
				},opt.model_dir)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

	np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy',losses)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy',ssims)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)

def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		pred=net(inputs)
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	return np.mean(ssims) ,np.mean(psnrs)


if __name__ == "__main__":
	if opt.trainset not in loaders_ or loaders_[opt.trainset] is None:
		raise ValueError(f'trainset "{opt.trainset}" is not available. Check --data_dir and folder layout.')
	if opt.testset not in loaders_ or loaders_[opt.testset] is None:
		raise ValueError(f'testset "{opt.testset}" is not available. Check --data_dir and folder layout.')

	loader_train=loaders_[opt.trainset]
	loader_test=loaders_[opt.testset]
	net=models_[opt.net]
	net=net.to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.perloss:
			vgg_model = vgg16(pretrained=True).features[:16]
			vgg_model = vgg_model.to(opt.device)
			for param in vgg_model.parameters():
				param.requires_grad = False
			criterion.append(PerLoss(vgg_model).to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net,loader_train,loader_test,optimizer,criterion)
