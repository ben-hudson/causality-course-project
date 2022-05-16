import torch
import cooper


def compute_nll(log_likelihood, valid, opt):
    valid = valid.type(log_likelihood.type())
    if opt.include_invalid:
        nll = - torch.mean(log_likelihood)
    else:
        nll = - torch.dot(log_likelihood, valid) / torch.sum(valid)

    return nll


class CustomCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, g_reg_coeff=0., gc_reg_coeff=0., g_constraint=0, gc_constraint=0., g_scaling=1., gc_scaling=1., linear_schedule=False, max_g=0, max_gc=0):
        self.is_constrained = (g_constraint > 0. or gc_constraint > 0.)
        self.g_reg_coeff = g_reg_coeff
        self.gc_reg_coeff = gc_reg_coeff
        self.g_constraint = g_constraint
        self.gc_constraint = gc_constraint

        if linear_schedule:
            self.g_effective_constraint = max_g
            self.gc_effective_constraint = max_gc
        else:
            self.g_effective_constraint = g_constraint
            self.gc_effective_constraint = gc_constraint

        self.max_g, self.max_gc = max_g, max_gc

        self.g_scaling = g_scaling
        self.gc_scaling = gc_scaling
        super().__init__(is_constrained=self.is_constrained)

    def closure(self, model, obs, cont_c, disc_c, valid, other, opt):
        misc = {}
        if opt.mode == "vae":
            elbo, reconstruction_loss, kl, _ = model.elbo(obs, cont_c, disc_c)
            loss = compute_nll(elbo, valid, opt)
        elif opt.mode == "supervised_vae":
            obs = obs[:, -1]
            other = other[:, -1]
            z_hat = model.latent_model.mean(model.latent_model.transform_q_params(model.encode(obs)))
            loss = torch.mean((z_hat.view(z_hat.shape[0], -1) - other) ** 2)
            reconstruction_loss, kl = 0, 0
        elif opt.mode == "latent_transition_only":
            ll = model.log_likelihood(other, cont_c, disc_c)
            loss = compute_nll(ll, valid, opt)
            reconstruction_loss, kl = 0, 0
        else:
            raise NotImplementedError(f"--mode {opt.mode} is not implemented.")

        misc["nll"] = loss.item()
        misc["reconstruction_loss"] = reconstruction_loss
        misc["kl"] = kl

        # regularization/constraint
        g_reg = model.latent_model.g_regularizer()
        gc_reg = model.latent_model.gc_regularizer()

        misc["g_reg"], misc["gc_reg"] = g_reg.item(), gc_reg.item()

        if not self.is_constrained:
            if self.g_reg_coeff > 0:
                loss += opt.g_reg_coeff * g_reg * g_scaling
            if self.gc_reg_coeff > 0:
                loss += opt.gc_reg_coeff * gc_reg * gc_scaling

            return cooper.CMPState(loss=loss, ineq_defect=None, eq_defect=None, misc=misc)
        else:
            defects = []
            if self.g_constraint > 0:
                defects.append(g_reg - self.g_effective_constraint)
            if self.gc_constraint > 0:
                defects.append(gc_reg - self.gc_effective_constraint)

            defects = torch.stack(defects)

            return cooper.CMPState(loss=loss, ineq_defect=defects, eq_defect=None, misc=misc)

    def update_constraint(self, iter, total_iter):
        if iter <= total_iter:
            if self.g_constraint > 0:
                self.g_effective_constraint = (self.max_g - self.g_constraint) * (1 - iter / total_iter) + self.g_constraint
            if self.gc_constraint > 0:
                self.gc_effective_constraint = (self.max_gc - self.gc_constraint) * (1 - iter / total_iter) + self.gc_constraint
        else:
            if self.g_constraint > 0:
                self.g_effective_constraint = self.g_constraint
            if self.gc_constraint > 0:
                self.gc_effective_constraint = self.gc_constraint

