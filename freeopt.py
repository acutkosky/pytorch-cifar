
import torch
from torch.optim import Optimizer

SMALL_VALUE = 1e-8

class FreeOpt(Optimizer):

    def __init__(self, params, lr, eps=1.0):

        defaults = dict(lr=lr, eps=eps)

        super().__init__(params, defaults)


    # def __setstate__(self, state):
    #     super().__setstate__(state)
    #     for group in self.param_groups:
    #         group.setdefault('nesterov', False)
    #         group.setdefault('maximize', False)
    #         group.setdefault('foreach', None)


    def get_alpha(self, eps, V, max_G, eta):
        V_plus = V + 4 * max_G**2
        # print(f"V_plus/max_G**2: {V_plus/max_G**2}, torch.log(V_plus/max_G**2): {torch.log(V_plus/max_G**2)}. V_plus: {V_plus}" )
        alpha = eps * max_G**2 / (V_plus * torch.log(V_plus/max_G**2)**2)
        # print(f"alpha: {alpha}")

        return eps*eta**2


    def get_theta(self, prev_p, alpha, eta):
        abs_prev_p = torch.abs(prev_p) + SMALL_VALUE
        theta = 2 * prev_p * torch.log(abs_prev_p / alpha + 1)/(eta * abs_prev_p)

        return theta
    
    @torch.no_grad()
    def step(self, closure=None):
        ######
        #####
        #####
        # NOTE THAT OUR EXPRESSION DOES SOME WEIRD STUFF BECUASE OF THE


        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:

                    grad = p.grad
                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        # Exponential moving average of gradient values
                        state['prev_offset'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['start'] = p.clone().detach_()
                        state['max_G'] = torch.full_like(p, SMALL_VALUE, memory_format=torch.preserve_format)
                        state['V'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    prev_offset = state['prev_offset']
                    max_G = state['max_G']
                    start = state['start']
                    V = state['V']



                    old_eta = lr

                    old_alpha = self.get_alpha(eps, V, max_G ,old_eta)
                    # old_eta = lr/torch.sqrt(V + max_G**2 + SMALL_VALUE) #/ max_G

                    theta = self.get_theta(prev_offset, old_alpha, old_eta)

                    # prev_offset.copy_(p - start).detach_()

                    abs_grad = torch.abs(grad)

                    clip_grad = torch.clip(grad, max=max_G, min=-max_G)
                    max_G.copy_(torch.maximum(max_G, abs_grad))



                    eta = lr #/ max_G
                    # eta = lr/torch.sqrt(V + max_G**2 + SMALL_VALUE)


                    V.add_(clip_grad**2)

                    theta.sub_(clip_grad)

                    abs_theta  = torch.abs(theta)
                    theta = torch.sign(theta) * torch.relu(abs_theta - 2 * eta * clip_grad**2)


                    abs_theta  = torch.abs(theta)
                    alpha = self.get_alpha(eps, V, max_G, eta)




 
                    new_offset = alpha * theta / (abs_theta + SMALL_VALUE) * (torch.exp( 0.5 * eta *abs_theta) - 1.0)

                    prev_offset.copy_(new_offset)

                    p.copy_(start + new_offset)

        return loss



class ScaleFree(torch.optim.Optimizer):

    def __init__(self, params, epsilon=1.0, constraint=1.0):
        super().__init__(params, {'epsilon': epsilon, 'constraint': constraint, 'lr': epsilon})
        self.__setstate__(self.state)

    def reset(self, epsilon=None, constraint=None, zero_center=False):
        for group in self.param_groups:
            if epsilon is not None:
                group['epsilon'] = epsilon
            if constraint is not None:
                group['constraint'] = constraint
        self.zero_center = zero_center
        self.__setstate__(self.state)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['hint'] = torch.full_like(param, SMALL_VALUE)
                state['V'] = torch.full_like(param, SMALL_VALUE)
                state['b_inc'] = torch.full_like(param, 4)
                state['B'] = torch.full_like(param, 16)
                state['D'] = torch.full_like(param, SMALL_VALUE)
                state['initial_param'] = param.clone().detach()


                # if self.zero_center:
                #     state['theta'] = self.get_theta(param, alpha, V, h)
                #     state['initial_param'] = torch.zeros_like(param)
                # else:
                state['theta'] = torch.zeros_like(param)
                state['initial_param'] = param.clone().detach()


    
    def get_theta(self, offset, alpha, V, h):
        abs_offset = torch.abs(offset)
        f_theta = torch.log(abs_offset / torch.clip(alpha, min=SMALL_VALUE) + 1)

        f_transition = V / h**2

        theta_low = (f_theta  + V/h**2) * 3*h
        theta_high = torch.sqrt(f_theta * 36 * V) 

        use_theta_low = (f_theta > f_transition)
        use_theta_high = 1.0 - use_theta_low

        theta = use_theta_low * theta_low + use_theta_high * theta_high

        return theta


    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:
            epsilon = group['epsilon']
            constraint = group['constraint']
            for param in group['params']:
                if param.grad is None:
                    continue

                state = self.state[param]

                h = state['hint']
                theta = state['theta']
                V = state['V']
                b_increment = state['b_inc']
                B = state['B']
                D = state['D']
                initial_param = state['initial_param']
                offset = param - initial_param

                grad = param.grad

                abs_grad = torch.abs(grad)
                
                trunc_grad = torch.clip(grad, -h, h)

                next_h = torch.maximum(h, abs_grad)

                D.add_(abs_grad/next_h)

                constraint_radius = torch.sqrt(D) * constraint

                constraint_grad = trunc_grad - trunc_grad * (torch.abs(offset) > constraint_radius) * (torch.sign(grad * offset) < 0)

                theta.add_(-constraint_grad)
                V.add_(torch.square(constraint_grad))
                b_increment.add_(torch.square(constraint_grad/h))
                B.add_(4*b_increment)
                h.copy_(next_h)

                alpha = epsilon/(torch.sqrt(B) * torch.square(torch.log(B)))

                f_branch_condition = (torch.abs(theta) * h) < (6 * V)

                f = (torch.square(theta)/(36 * V)) * f_branch_condition + (torch.abs(theta)/(3 * h) - V/torch.square(h)) * (torch.logical_not(f_branch_condition))

                next_offset  = alpha * torch.sign(theta) * (torch.exp(f) - 1.0)


                # print('initial_param: ', initial_param)
                # print('next_offset: ', next_offset)
                # print('theta: ', theta)
                # print('alpha: ', alpha)
                # print('expf -1: ', torch.exp(f)-1.0)
                # print('f :', f)
                # print('branch: ', f_branch_condition)
                # print('grad: ', grad)
                # print('trunc grad: ', trunc_grad)
                # print('constraint grad: ', constraint_grad)
                

                param.copy_(initial_param + next_offset)




if __name__ == '__main__':

    import numpy as np
    w = torch.tensor(np.ones(1), requires_grad = True)
    
    def loss(w):
        return torch.sum(w**2)

    opt = FreeOpt([w], lr=0.1, eps=1.0)
    opt = ScaleFree([w])
    for i in range(1000):
        l = loss(w)

        print('')
        print('iter: ', i)
        print('w: ', w)
        print('l: ', l)
        opt.zero_grad()
        l.backward()
        opt.step()
    