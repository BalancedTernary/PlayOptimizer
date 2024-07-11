import math
import torch
from torch.optim.optimizer import Optimizer

class Play(Optimizer):
    def __init__(self, params, lr=1e-4, smooth=10, point=0.9, soft_start=None, weight_decay=0, decay_flexibility=1, smoots_decrement=1e-2, snapshot_recovery_threshold=100, limit_snapshot_cycle=10, eps=1e-18):
        #parameter checks
        if lr<0: #学习速率
            raise ValueError(f'Invalid 学习速率: {lr}')
        if smooth is not None and smooth<0: #平滑程度
            raise ValueError(f'Invalid 平滑程度: {smooth}')
        if point<=0 or point>=1: #梯度稳定度调定点，值越大越积极降低学习率以稳定梯度.过大的值导致学习过早停止,过小的值导致参数无法收敛到固定值(只是会有极小波动,并非发散).建议取值在0.5~0.9之间
            raise ValueError(f'Invalid 梯度稳定度调定点: {point}')
        if soft_start is not None and (soft_start<=0): #软起动,这个参数将乘在lr上.初始值越小，启动越软
            raise ValueError(f'Invalid 软起动: {soft_start}')
        if weight_decay<0 or weight_decay>=1: #权重衰减
            raise ValueError(f'Invalid 权重衰减: {weight_decay}')
        if decay_flexibility<0: #权重衰减柔性
            raise ValueError(f'Invalid 权重衰减柔性: {decay_flexibility}')
        if smoots_decrement<0: #平滑衰减量,对loss取包络时使用,约小越平滑
            raise ValueError(f'Invalid 平滑衰减量: {smoots_decrement}')
        if snapshot_recovery_threshold<=1: #快照恢复阈值,当前平滑loss大于平滑loss的最小值的这个倍数时恢复快照
            raise ValueError(f'Invalid 快照恢复阈值: {snapshot_recovery_threshold}')
        if limit_snapshot_cycle<1: #快照周期下限
            raise ValueError(f'Invalid 快照周期下限: {limit_snapshot_cycle}')
        if eps<=0 or eps>=1: 
            raise ValueError(f'Invalid eps:{eps}')

        if soft_start is None:
            soft_start=smooth

        defaults = dict(lr=lr, smooth=smooth, point=point, soft_start=soft_start, weight_decay=weight_decay, decay_flexibility=decay_flexibility, smoots_decrement=smoots_decrement, snapshot_recovery_threshold=snapshot_recovery_threshold, limit_snapshot_cycle=limit_snapshot_cycle, eps=eps)
        super(Play, self).__init__(params,defaults)

        self.lr = lr
        self.smooth = smooth
        self.point=point
        self.soft_start=soft_start
        self.weight_decay = weight_decay
        self.decay_flexibility=decay_flexibility
        self.smoots_decrement=smoots_decrement
        self.snapshot_recovery_threshold=snapshot_recovery_threshold
        self.limit_snapshot_cycle=limit_snapshot_cycle
        self.upperEnvelope = -2**15
        self.lowerEnvelope = 2**15
        self.upperEnvelopeCache = -2**15
        self.lowerEnvelopeCache = 2**15
        self.minNeutral = 2**15 #上包络与下包络的平均值的最小值
        self.t=0 #计数器,拍摄快照后清零
        self.eps=eps
        

    def __setstate__(self, state):
        super(Play, self).__setstate__(state)

    def step(self, closure=None, loss=None):
        if closure is not None:
            loss = closure()
            loss.backward()
        if loss is None:
            loss=torch.tensor(1)
        
        pp=-math.log(2)/math.log(self.point)
        s=min((1+self.t)/(1+self.soft_start),1)
        k=self.lr*s
        
        
        self.t+=1

        self.upperEnvelope=self.upperEnvelope-self.smoots_decrement
        self.lowerEnvelope=self.lowerEnvelope+self.smoots_decrement
        if loss>self.upperEnvelope:
            self.upperEnvelope=float(loss)
        if loss<self.lowerEnvelope: #前面不能加else,不然因为两者的自动衰减会导致值出现错误
            self.lowerEnvelope=float(loss)
        neutral=(self.upperEnvelope+self.lowerEnvelope)/2
                        
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    p.data[~torch.isfinite(p.data)]=0
                    if p.grad is None:
                        continue
                    p.grad.data[~torch.isfinite(p.grad.data)]=0
                    if p.grad.data.is_sparse:
                        raise RuntimeError('PlayOptimizer does not support sparse gradients')
                    
                    state = self.state[p]  #get state dict for this param
                    if len(state) == 0:   #if first time to run...init dictionary with our desired entries
                        #state['sinhFlat'] = torch.zeros_like(p.grad.data)
                        #state['coshFlat'] = torch.ones_like(p.grad.data)
                        state['flat'] = torch.zeros_like(p.grad.data)
                        state['absFlat'] = torch.zeros_like(p.grad.data)
                        state['snapshot'] = p.data.clone()
                    
                    if self.t>=self.limit_snapshot_cycle and neutral<self.minNeutral: #不能在这里清零t,否则快照拍摄不完整
                        state['snapshot'] = p.data.clone()

                    #state['sinhFlat']=(state['sinhFlat']*self.smooth+p.grad.data.sinh())/(self.smooth+1)
                    #state['sinhFlat'][~torch.isfinite(state['sinhFlat'])]=0
                    #state['coshFlat']=(state['coshFlat']*self.smooth+p.grad.data.cosh())/(self.smooth+1)
                    #state['coshFlat'][~torch.isfinite(state['coshFlat'])]=0
                    nap=(p.grad.data.abs()/(state['flat'].abs()+self.eps)).clamp(min=0,max=1)
                    nap[p.grad.data.sign()*state['flat'].sign()<0]=1
                    state['flat']=(state['flat']*self.smooth+p.grad.data*nap)/(self.smooth+nap)
                    state['flat'][~torch.isfinite(state['flat'])]=0
                    state['absFlat']=(state['absFlat']*self.smooth+p.grad.data.abs()*nap)/(self.smooth+nap)
                    state['absFlat'][~torch.isfinite(state['absFlat'])]=0
                    
                    #d=-state['sinhFlat'].asinh()/state['coshFlat'].acosh()*(p.data.abs()*k+b)
                    n=state['flat'].abs()/(state['absFlat'])
                    n=(1-(1-n**2).sqrt())/n
                    #n=(1-(1-n**2).sqrt())
                    n[~torch.isfinite(n)]=0
                    n.clamp_(min=0,max=1)
                    n**=pp
                    n.clamp_(max=n.mean())
                    d=-state['flat'].sign()*n*k
                    
                    #对使参数远离0的训练,进行抑制,实现权重衰减.
                    decayMask=d*p.data>0
                    x=d[decayMask]
                    d[decayMask]+=(-2*self.weight_decay*x*(x/self.decay_flexibility).atan().abs())/math.pi
                    d[~torch.isfinite(d)]=0

                    p.data+=d

                    if neutral>self.snapshot_recovery_threshold*self.minNeutral or not loss.isfinite():
                        #state['sinhFlat'] = torch.zeros_like(p.grad.data)
                        #state['coshFlat'] = torch.ones_like(p.grad.data)
                        state['flat'] = torch.zeros_like(p.grad.data)
                        state['absFlat'] = torch.zeros_like(p.grad.data)
                        p.grad.data = torch.zeros_like(p.grad.data)
                        p.data=state['snapshot'].clone()
                        
                   
        if self.t>=self.limit_snapshot_cycle and neutral<self.minNeutral:
            self.t=0
        if neutral<self.minNeutral:
            self.minNeutral=float(neutral)
        if neutral>self.snapshot_recovery_threshold*self.minNeutral or not loss.isfinite():
            self.t=0
            self.upperEnvelope = float(self.upperEnvelopeCache)
            self.lowerEnvelope = float(self.lowerEnvelopeCache)

        self.upperEnvelopeCache = float(self.upperEnvelope)
        self.lowerEnvelopeCache = float(self.lowerEnvelope)

        return loss



