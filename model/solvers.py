import torch

def power_method(A, b, num_iter=1000, tol=1e-6, verbose=True):
	"""
	Power method for operator pytorch operator A and initial vector b.
	"""
	eig_old = torch.zeros(1)
	flag_tol_reached = False
	for it in range(num_iter):
		b = A(b)
		b = b / torch.norm(b)
		eig_max = torch.sum(b*A(b))
		if verbose:
			print('i:{0:3d} \t |e_new - e_old|:{1:2.2e}'.format(it,abs(eig_max-eig_old).item()))
		if abs(eig_max-eig_old)<tol:
			flag_tol_reached = True
			break
		eig_old = eig_max
	if verbose:
		print('tolerance reached!',it)
		print(f"L = {eig_max.item():.3e}")
	return eig_max.item(), b, flag_tol_reached

def uball_project(W, dim=(2,3)):
    """ projection of W onto the unit ball
    """
    normW = torch.norm(W, dim=dim, keepdim=True)
    return W * torch.clamp(1/normW, max=1)

