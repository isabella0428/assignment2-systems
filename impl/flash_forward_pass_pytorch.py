import torch
import numpy as np

def ceil_div(N, D):
    return (N + D - 1) // D

def pad(x, expect_rows):
	if x.shape[-2] < expect_rows:
		pad_rows = expect_rows - x.shape[-2]
		x = torch.nn.functional.pad(x, (0, 0, 0, pad_rows))  # pad bottom
	return x 

class FlashForwardPassPytorch(torch.autograd.Function):
	"""
	Your implementation should take input Q, K, and V as well as a flag is_causal and produce
	the output O and the logsumexp value L. You can ignore the is_causal flag for this task. The
	autograd.Function forward should then use save L, Q, K, V, O for the backward pass and
	return O. Remember that the implementation of the forward method of autograd.Function
	always takes the context as its first parameter.
	"""
	def forward(ctx, Q, K, V, is_casual=False):
		*batch_dims, N_q, d = Q.shape
		*batch_dims, N_k, d = K.shape

		O = torch.zeros(*batch_dims, N_q, d)
		L = torch.zeros(*batch_dims, N_q, 1)

		B_q = 16					# Q matrix tile size
		T_q = ceil_div(N_q, B_q)	# Q matrix: Number of tiles

		B_k = 16					# K matrix tile size
		T_k = ceil_div(N_k, B_k)	# K matrix tile size
	
		for i in range(1, T_q + 1):
			start = (i-1) * B_q
			end = min(i*B_q, N_q)
			Q_i = pad(Q[..., start:end, :], B_q)			# B_q * d

			O_i_j = torch.zeros(*batch_dims, B_q, d)
			l_i_j = torch.zeros(*batch_dims, B_q, 1)
			m_i_j = torch.ones(*batch_dims, B_q, 1) * -np.inf

			for j in range(1, T_k + 1):
				start = (j-1) * B_k
				end = j * B_k
				K_j = pad(K[..., start:end, :], B_k)		# B_k * d
				V_j = pad(V[..., start:end, :], B_k)		# B_k * d

				S_i_j = Q_i @ K_j.transpose(-2, -1) / np.sqrt(d)
				new_m_i_j = torch.max(m_i_j, torch.amax(S_i_j, dim=-1, keepdim=True))

				P_local_i_j = torch.exp(S_i_j - new_m_i_j)

				alpha = torch.exp(m_i_j - new_m_i_j)  # Shape: (..., B_q, 1)
				l_i_j = alpha * l_i_j + torch.sum(P_local_i_j, dim = -1, keepdim=True)		# Sum of e^{x - m(x)}
				O_i_j = alpha * O_i_j + P_local_i_j @ V_j		# P * V

				m_i_j = new_m_i_j
			
			O_i = O_i_j / l_i_j							# Divide the Sum of e^{x - m(x)} to get softmax value
			L_i = m_i_j + np.log(l_i_j)					# Get log: recover to the state log (sum (exp (S_i_j))) 

			start = (i-1) * B_q
			end = min(i*B_q, N_q)
			O[..., start:end, :] = O_i
			L[..., start:end, :] = L_i
		

		L = L.squeeze(-1)
		ctx.save_for_backward(Q, K, V, O, L)
		return O


	def backward():
		raise NotImplementedError
