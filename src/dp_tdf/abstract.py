from abc import ABCMeta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.utils.utils import sdr, simplified_msseval

def _check_finite_tensor(name, x):
    if x is None:
        return
    if not torch.is_tensor(x):
        return
    if not torch.isfinite(x).all():
        nan_cnt = torch.isnan(x).sum().item()
        inf_cnt = torch.isinf(x).sum().item()

        x_safe = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)

        print("\n" + "=" * 80)
        print(f"[NON-FINITE DETECTED] {name}")
        print(f"shape = {tuple(x.shape)}")
        print(f"nan_count = {nan_cnt}")
        print(f"inf_count = {inf_cnt}")
        print(
            f"safe_min = {x_safe.min().item():.6f}, "
            f"safe_max = {x_safe.max().item():.6f}, "
            f"safe_mean = {x_safe.mean().item():.6f}, "
            f"safe_std = {x_safe.std().item():.6f}"
        )
        print("=" * 80 + "\n")
        raise RuntimeError(f"Non-finite tensor detected in {name}")

class AbstractModel(LightningModule):
	__metaclass__ = ABCMeta

	def __init__(self, target_name,
				 lr, optimizer,
				  dim_f, dim_t, n_fft, n_fft_short, n_fft_long, hop_length, overlap,
				 audio_ch,
				  **kwargs):
		super().__init__()
		self.target_name = target_name	#当前模型要分离的目标源名字 如vocal、drums等
		self.lr = lr	#学习率learning rate
		self.optimizer = optimizer	#优化器种类 如Adam
		self.dim_c_in = audio_ch * 2	#输入通道数（*2是因为短时傅里叶之后包含实部和虚部：左声道实部&左声道虚部&右声道实部&右声道虚部）
		self.dim_c_out = audio_ch * 2	#输出通道数
		self.dim_f = dim_f	#模型真正保留的频率维大小（通常小于n_bins）
		self.dim_t = dim_t	#时间维长度  单位：帧

		self.n_fft_mid = n_fft	#中窗长
		self.n_fft_short = n_fft_short	#短窗长
		self.n_fft_long = n_fft_long	#长窗长
		self.n_fft = self.n_fft_mid  # STFT窗长（每个STFT窗口的采样点数量）

		self.n_bins_mid = self.n_fft_mid // 2 + 1
		self.n_bins_short = self.n_fft_short // 2 + 1
		self.n_bins_long = self.n_fft_long // 2 + 1
		self.n_bins = self.n_bins_mid  # 有效的频率点个数（奈奎斯特频率） stft之后的max频率

		self.hop_length = hop_length	#滑窗每次往前走多少（两帧之间采样点数量）
		self.audio_ch = audio_ch	#声道数

		#chunk：一段连续的音频片段，它是模型实际处理的输入单元。（切割原信号为很多小片）
		self.chunk_size = hop_length * (self.dim_t - 1)	#训练的chunk，单次数据量。多个chunk组成一个batch
		self.inference_chunk_size = hop_length * (self.dim_t*2 - 1)	#比训练的chunk更长，希望看到更多一点上下文
		self.overlap = overlap	# 推理时分块之间的重叠长度

		# 固定Hann窗，给STFT/ISTFT用，不参与梯度更新（requires_grad=False）
		#self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
		self.window_mid = nn.Parameter(torch.hann_window(window_length=self.n_fft_mid, periodic=True),requires_grad=False)
		self.window_short = nn.Parameter(torch.hann_window(window_length=self.n_fft_short, periodic=True),requires_grad=False)
		self.window_long = nn.Parameter(torch.hann_window(window_length=self.n_fft_long, periodic=True),requires_grad=False)
		self.window = self.window_mid

		self.freq_pad = nn.Parameter(torch.zeros([1, self.dim_c_out, self.n_bins - self.dim_f, 1]), requires_grad=False)
		#固定的0张量，istft时把模型输出的裁剪后频谱补回到完整频谱纬度（dim_f->n_bins）。因为模型输出只预测中窗域前 dim_f 个频率 bin，所以 istft() 前需要补回完整的中窗频率维
		self.inference_chunk_shape = (self.stft(torch.zeros([1, audio_ch, self.inference_chunk_size]))).shape
		#初始化时，先拿一个全0的假输入过一次stft，把推理chunk对应的频谱shape存下来，提前知道推理时一块音频进频谱后长什么样。


	def configure_optimizers(self):
		# 根据配置决定训练时用哪种优化器
		if self.optimizer == 'rmsprop':
			print("Using RMSprop optimizer")
			return torch.optim.RMSprop(self.parameters(), self.lr)
		elif self.optimizer == 'adamW':
			print("Using AdamW optimizer")
			return torch.optim.AdamW(self.parameters(), self.lr)

	def comp_loss(self, pred_detail, target_wave):
		# 把模型输出从频谱域转回波形域，再计算 L1 损失。
		_check_finite_tensor("comp_loss/pred_detail_before_istft", pred_detail)
		_check_finite_tensor("comp_loss/target_wave", target_wave)

		pred_detail = self.istft(pred_detail)  # 模型输出的是频谱，先ISTFT回波形
		_check_finite_tensor("comp_loss/pred_detail_after_istft", pred_detail)

		comp_loss = F.l1_loss(pred_detail, target_wave)  # 用波形域的 L1 loss 做训练目标
		_check_finite_tensor("comp_loss/value", comp_loss)

		self.log("train/comp_loss", comp_loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=False)  # 把损失记录到日志

		return comp_loss

	def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
		# 定义训练时一个batch怎么跑。*args, **kwargs 用于接收框架自动传入的参数。
		# 通常 Lightning 调用时会传入 (batch, batch_idx)。这里通过 args[0] 获取第一个参数，即 batch。
		mix_wave, target_wave = args[0]  # (batch, c, 261120)

		_check_finite_tensor("train/mix_wave", mix_wave)
		_check_finite_tensor("train/target_wave", target_wave)

		# input 1
		mix_specs = self.multi_stft(mix_wave)

		if isinstance(mix_specs, dict):
			for k, v in mix_specs.items():
				_check_finite_tensor(f"train/mix_specs[{k}]", v)
		else:
			_check_finite_tensor("train/mix_specs", mix_specs)

		# forward
		t_est_stft = self(mix_specs)  # (batch, c, 1044, 256)
		_check_finite_tensor("train/t_est_stft", t_est_stft)

		loss = self.comp_loss(t_est_stft, target_wave)
		_check_finite_tensor("train/loss", loss)

		self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)

		return {"loss": loss}


	# Validation SDR is calculated on whole tracks and not chunks since
	# short inputs have high possibility of being silent (all-zero signal)
	# which leads to very low sdr values regardless of the model.
	# A natural procedure would be to split a track into chunk batches and
	# load them on multiple gpus, but aggregation was too difficult.
	# So instead we load one whole track on a single device (data_loader batch_size should always be 1)
	# and do all the batch splitting and aggregation on a single device.

	def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:  # 验证集上按照整首歌的方式评估模型
		mix_chunk_batches, target = args[0]  # mix_chunk_batches是一个列表，每个其中的每个元素是一个batch

		# remove data_loader batch dimension
		# [(b, c, time)], (c, all_times)
		mix_chunk_batches, target = [batch[0] for batch in mix_chunk_batches], target[0]

		_check_finite_tensor("val/target", target)

		# process whole track in batches of chunks
		target_hat_chunks = []
		for i, batch in enumerate(mix_chunk_batches):
			_check_finite_tensor(f"val/batch[{i}]", batch)

			# input
			mix_specs = self.multi_stft(batch)  # (batch, c*2, 1044, 256)

			if isinstance(mix_specs, dict):
				for k, v in mix_specs.items():
					_check_finite_tensor(f"val/mix_specs[{i}][{k}]", v)
			else:
				_check_finite_tensor(f"val/mix_specs[{i}]", mix_specs)

			pred_detail = self(mix_specs)  # (batch, c, 1044, 256), irm
			_check_finite_tensor(f"val/pred_detail_stft[{i}]", pred_detail)

			pred_detail = self.istft(pred_detail)  # 得到每个chunk的波形
			_check_finite_tensor(f"val/pred_detail_wave[{i}]", pred_detail)

			target_hat_chunks.append(pred_detail[..., self.overlap:-self.overlap])  # 减少chunk边界伪影，存入target_hat_chunks

		target_hat_chunks = torch.cat(target_hat_chunks)  # (b*len(ls),c,t) 拼接（总块数，c，有效长度）
		_check_finite_tensor("val/target_hat_chunks_cat", target_hat_chunks)

		# concat all output chunks (c, all_times)
		target_hat = target_hat_chunks.transpose(0, 1).reshape(self.audio_ch, -1)[
			..., :target.shape[-1]]  # 交换前两维，后两维合并，截取与目标相同的长度
		_check_finite_tensor("val/target_hat", target_hat)

		ests = target_hat.detach().cpu().numpy()  # (c, all_times)
		references = target.cpu().numpy()

		if not np.isfinite(ests).all():
			raise RuntimeError("Non-finite ests detected before SDR")
		if not np.isfinite(references).all():
			raise RuntimeError("Non-finite references detected before SDR")

		score = sdr(ests, references)

		# (src, t, c)
		SDR = simplified_msseval(np.expand_dims(references.T, axis=0), np.expand_dims(ests.T, axis=0), chunk_size=44100)

		if not np.isfinite(score):
			raise RuntimeError(f"Non-finite val song SDR detected: {score}")

		return {'song': score, 'chunk': SDR}

	def validation_epoch_end(self, outputs) -> None:  # 把整轮验证里所有歌曲的结果汇总，得到最终验证指标。
		songs = torch.Tensor([x['song'] for x in outputs])
		_check_finite_tensor("val_epoch/songs", songs)

		avg_uSDR = songs.mean()  # 把每首歌的 song-level SDR 求平均
		_check_finite_tensor("val_epoch/avg_uSDR", avg_uSDR)

		self.log("val/usdr", avg_uSDR, sync_dist=True, on_step=False, on_epoch=True, logger=True)

		chunks = [x['chunk'][0, :] for x in outputs]
		# concat np array
		chunks = np.concatenate(chunks, axis=0)

		if not np.isfinite(chunks).any():
			raise RuntimeError("All chunk SDR values are non-finite in validation_epoch_end")

		median_cSDR = np.nanmedian(chunks.flatten(), axis=0)
		# 把所有 chunk 的 SDR 拼起来，取中位数cSDR
		median_cSDR = float(median_cSDR)

		if not np.isfinite(median_cSDR):
			raise RuntimeError(f"Non-finite median_cSDR detected: {median_cSDR}")

		self.log("val/csdr", median_cSDR, sync_dist=True, on_step=False, on_epoch=True, logger=True)

	def _stft_impl(self, x, n_fft, window):
		"""
        通用STFT实现
        输入x: (B, C, T)
        输出(B, C*2, F, T_frames)
        """
		dim_b = x.shape[0]
		x = x.reshape([dim_b * self.audio_ch, -1])
		x = torch.stft(
			x,
			n_fft=n_fft,
			hop_length=self.hop_length,
			window=window,
			center=True,
			return_complex=True,
		)
		x = torch.view_as_real(x)
		x = x.permute([0, 3, 1, 2])
		x = x.reshape([dim_b, self.audio_ch, 2, x.shape[-2], -1]).reshape(
			[dim_b, self.audio_ch * 2, x.shape[-2], -1]
		)

		return x

	def stft(self, x):
		"""
        为了兼容原始代码，stft默认仍然表示中窗STFT
        输出频率维仍然裁到self.dim_f，作为主干输入域
        """
		x = self._stft_impl(x, self.n_fft_mid, self.window_mid)
		return x[:, :, :self.dim_f]

	def stft_short(self, x):
		"""
        短窗 STFT
        第一版不裁频率维，后面交给前端模块统一对齐
        """
		return self._stft_impl(x, self.n_fft_short, self.window_short)

	def stft_long(self, x):
		"""
        长窗 STFT
        第一版不裁频率维，后面交给前端模块统一对齐
        """
		return self._stft_impl(x, self.n_fft_long, self.window_long)

	def multi_stft(self, x):
		"""
        返回三路输入
        约定：
        - mid是主干输入域
        - short / long 作为辅助前端输入
        """
		return {
			"short": self.stft_short(x),	#待裁剪
			"mid": self.stft(x),  # 中窗，保留原始 DTT 输入域
			"long": self.stft_long(x),	#待裁剪
		}

	def istft(self, x):
		'''
		Args:
		x: (batch, c*2, 2048, 256)
		'''
		dim_b = x.shape[0]

		x = torch.cat(
		[x, self.freq_pad.repeat([x.shape[0], 1, 1, x.shape[-1]])],
		-2
		)  # (batch, c*2, 3073, 256)

		x = x.reshape([dim_b, self.audio_ch, 2, self.n_bins, -1]).reshape(
		[dim_b * self.audio_ch, 2, self.n_bins, -1]
		)  # (batch*c, 2, 3073, 256)

		x = x.permute([0, 2, 3, 1]).contiguous()  # (batch*c, 3073, 256, 2)
		x = torch.view_as_complex(x)  # (batch*c, 3073, 256)

		x = torch.istft(
		x,
		n_fft=self.n_fft,
		hop_length=self.hop_length,
		window=self.window,
		center=True,
		)  # (batch*c, 261120)

		return x.reshape([dim_b, self.audio_ch, -1])  # (batch, c, 261120)

	def demix(self, mix, inf_chunk_size, batch_size=5, inf_overf=4):
		'''
		Args:
			mix: (C, L)
		Returns:
			est: (src, C, L)
		'''

		# batch_size = self.config.inference.batch_size
		#  = self.chunk_size
		# self.instruments = ['bass', 'drums', 'other', 'vocals']
		num_instruments = 1

		inf_hop = inf_chunk_size // inf_overf  # hop size
		L = mix.shape[1]
		pad_size = inf_hop - (L - inf_chunk_size) % inf_hop
		mix = torch.cat([torch.zeros(2, inf_chunk_size - inf_hop), torch.Tensor(mix), torch.zeros(2, pad_size + inf_chunk_size - inf_hop)], 1)
		mix = mix.cuda()

		chunks = []
		i = 0
		while i + inf_chunk_size <= mix.shape[1]:
			chunks.append(mix[:, i:i + inf_chunk_size])
			i += inf_hop
		chunks = torch.stack(chunks)

		batches = []
		i = 0
		while i < len(chunks):
			batches.append(chunks[i:i + batch_size])
			i = i + batch_size

		X = torch.zeros(num_instruments, 2, inf_chunk_size - inf_hop) # (src, c, t)
		X = X.cuda()
		with torch.cuda.amp.autocast():
			with torch.no_grad():
				for batch in batches:
					x = self.stft(batch)
					x = self(x)
					x = self.istft(x) # (batch, c, 261120)
					# insert new axis, the model only predict 1 src so we need to add axis
					x = x[:,None, ...] # (batch, 1, c, 261120)
					x = x.repeat([ 1, num_instruments, 1, 1]) # (batch, src, c, 261120)
					for w in x: # iterate over batch
						a = X[..., :-(inf_chunk_size - inf_hop)]
						b = X[..., -(inf_chunk_size - inf_hop):] + w[..., :(inf_chunk_size - inf_hop)]
						c = w[..., (inf_chunk_size - inf_hop):]
						X = torch.cat([a, b, c], -1)

		estimated_sources = X[..., inf_chunk_size - inf_hop:-(pad_size + inf_chunk_size - inf_hop)] / inf_overf

		assert L == estimated_sources.shape[-1]

		return estimated_sources

