# AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
# 来源：https://github.com/QingruZhang/AdaLoRA

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List

class LoRALayer:
    """LoRA层的基类"""
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class SVDLinear(nn.Linear, LoRALayer):
    """
    AdaLoRA中基于SVD的适配线性层
    
    AdaLoRA使用SVD分解来实现自适应秩分配：
    W = W₀ + α * B * E * A
    其中E是对角奇异值矩阵，可以动态调整秩
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # 可训练参数
        if r > 0:
            # A矩阵：r × in_features
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            # E矩阵：奇异值对角矩阵，形状为r × 1
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            ) 
            # B矩阵：out_features × r
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            # 记录当前有效秩数
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)   
            # 冻结预训练权重矩阵
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        """重置参数"""
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # 初始化A,B与nn.Linear默认方式相同
            # E（奇异值）初始化为零
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        """训练模式：解除权重合并"""
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # 确保权重没有合并
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum + 1e-5)
            self.merged = False

    def eval(self):
        """评估模式：合并权重"""
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # 合并权重并标记
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum + 1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        """前向传播"""
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                # AdaLoRA的SVD计算：x @ (A * E)^T @ B^T
                result += (
                    self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                ) * self.scaling / (self.ranknum + 1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class RankAllocator(object):
    """
    AdaLoRA的秩分配器，在每个训练步骤中调用
    
    自适应预算分配算法：
    1. 计算重要性分数（sensitivity + uncertainty）
    2. 根据预算调度器确定目标秩
    3. 掩码不重要的奇异值
    
    Args:
        model: 应用AdaLoRA的模型
        lora_r: 每个增量矩阵的初始秩
        target_rank: 增量矩阵的目标平均秩
        init_warmup: 初始微调预热步数
        final_warmup: 最终微调步数
        mask_interval: 两次预算分配之间的时间间隔
        beta1: 敏感性平滑的EMA超参数
        beta2: 不确定性量化的EMA超参数
        total_step: 总训练步数
        target_total_rank: 指定的最终总秩
        tb_writter: Tensorboard SummaryWriter
        tb_writter_loginterval: SummaryWriter的日志间隔
    """
    def __init__(
        self, model, 
        lora_r: int,
        target_rank: int, 
        init_warmup: int, 
        final_warmup: int,
        mask_interval: int,
        beta1: float, 
        beta2: float, 
        total_step: Optional[int] = None, 
        target_total_rank: Optional[int] = None,
        tb_writter=None,
        tb_writter_loginterval: int = 500, 
    ):
        self.ave_target_rank = target_rank 
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r 
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {}  # importance scores
        self.exp_avg_ipt = {}  # exponential moving average of importance
        self.exp_avg_unc = {}  # exponential moving average of uncertainty
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 

        assert (self.beta1 < 1 and self.beta1 > 0)
        assert (self.beta2 < 1 and self.beta2 > 0)

    def set_total_step(self, total_step: int): 
        """设置总步数"""
        self.total_step = total_step
        assert self.total_step > self.initial_warmup + self.final_warmup

    def get_rank_pattern(self):
        """返回秩模式"""
        return self.rank_pattern

    def get_lora_param_name(self):
        """准备预算调度器"""
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n, p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 

    def schedule_threshold(self, step: int):
        """全局预算调度"""
        mask_ind = False 
        target_rank = self.target_rank 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        
        if step <= initial_warmup: 
            # 初始预热
            curr_rank = self.total_rank 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            # 最终微调
            curr_rank = self.target_rank 
            # 通过始终掩码相同的不重要奇异值来固定秩模式
            mask_ind = True 
        else: 
            # 预算递减
            mul_coeff = 1 - (step - initial_warmup) / (total_step - final_warmup - initial_warmup)
            curr_rank = target_rank + (self.total_rank - target_rank) * (mul_coeff ** 3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False 
        return curr_rank, mask_ind 

    def update_ipt(self, model): 
        """更新重要性分数"""
        for n, p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # 计算敏感性 
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # 更新敏感性
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1 - self.beta1) * self.ipt[n]
                    # 更新不确定性
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n, p=None, metric="ipt"):
        """计算重要性分数"""
        if metric == "ipt":
            # 结合敏感性和不确定性
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexpected Metric: %s" % metric)
        return ipt_score 

    def _combine_ipt(self, ipt_E, ipt_AB):
        """组合重要性分数"""
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank): 
        """掩码到目标秩"""
        is_dict = {}
        combine_dict = {} 
        singular_dict = {}
        
        # 计算每个子矩阵的重要性分数
        for n, p in model.named_parameters(): 
            if "lora_A" in n: 
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n: 
                hdim_b, rdim = p.shape 
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")                
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        # 组合重要性分数
        all_is = []
        for name_mat in combine_dict: 
            ipt_E = singular_dict[name_mat] 
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat % "lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

        # 计算掩码阈值
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank - curr_rank))[0].item()

        # 掩码不重要的奇异值
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n, p in model.named_parameters():
                if "lora_E" in n: 
                    p.data.masked_fill_(is_dict[n] <= mask_threshold, 0.0)
                    ranknum = (is_dict[n] > mask_threshold).sum().item() 

                    if self.tb_writter is not None and self.global_step % self.log_interval == 0:
                        self.tb_writter.add_scalar("Ranknum/%s" % (n,), ranknum, self.global_step) 
                        self.rank_pattern[n] = ranknum 
                        curr_sum_rank += ranknum 
                        sum_param += ranknum * self.shape_dict[n.replace("lora_E", "lora_A")][1]  
                        sum_param += ranknum * self.shape_dict[n.replace("lora_E", "lora_B")][0]  

            if self.tb_writter is not None and self.global_step % self.log_interval == 0:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return mask_threshold

    def update_and_mask(self, model, global_step):
        """更新重要性分数并应用掩码"""
        if global_step < self.total_step - self.final_warmup:
            # 逐元素更新重要性分数
            self.update_ipt(model)
            # 在最终微调期间不更新ipt
        
        # 预算调度
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # 掩码到目标预算
            mask_threshold = self.mask_to_target_rank(model, curr_rank) 
        else:
            mask_threshold = None 
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        """可能的tensorboard日志记录"""
        if self.tb_writter is not None and self.global_step % self.log_interval == 0:
            with torch.no_grad():
                regu_loss = []
                for n, p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov - I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s" % n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss) / len(regu_loss), self.global_step
                )

def compute_orth_regu(model, regu_weight=0.1):
    """
    计算模型中SVDLinear的正交正则化
    
    Args:
        model: 包含SVDLinear层的模型
        regu_weight: 正则化权重
    
    Returns:
        正交正则化损失
    """
    regu_loss, num_param = 0., 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1
    return regu_weight * regu_loss / num_param 