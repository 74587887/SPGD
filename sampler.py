import tqdm
import torch
import numpy as np
import torch.nn as nn


def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError
    return RUN(**kwargs)


class Scheduler(nn.Module):
    """
        Scheduler for diffusion sigma(t) and discretization step size Delta t
    """

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear',
                 timestep='poly-7'):
        """
            Initializes the scheduler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the schedule.
                sigma_max (float): Maximum value of sigma.
                sigma_min (float): Minimum value of sigma.
                sigma_final (float): Final value of sigma, defaults to sigma_min.
                schedule (str): Type of schedule for sigma ('linear' or 'sqrt').
                timestep (str): Type of timestep function ('log' or 'poly-n').
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = self.sigma_min
        self.schedule = schedule
        self.timestep = timestep

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)

        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])

        # factor = 2\dot\sigma(t)\sigma(t)\Delta t
        factor_steps = np.array(
            [2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps, self.time_steps, self.factor_steps = sigma_steps, time_steps, factor_steps
        self.factor_steps = [max(f, 0) for f in self.factor_steps]

    def get_sigma_fn(self, schedule):
        """
            Returns the sigma function, its derivative, and its inverse based on the given schedule.
        """
        if schedule == 'sqrt':
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma ** 2

        elif schedule == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        """
            Returns the time step function based on the given timestep type.
        """
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        else:
            raise NotImplementedError
        return get_time_step_fn


class DiffusionSampler(nn.Module):
    """
        Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__()
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, SDE=False, record=False, verbose=False):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, SDE, record, verbose)
        else:
            raise NotImplementedError

    def _euler(self, model, x_start, SDE=False, record=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step]
            score = model.score(x, sigma)
            if SDE:
                epsilon = torch.randn_like(x)
                x = x + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x + factor * score * 0.5
            # record
            if record:
                if SDE:
                    self._record(x, score, sigma, factor, epsilon)
                else:
                    self._record(x, score, sigma, factor)
        return x

    def _record(self, x, score, sigma, factor, epsilon=None):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', x)
        self.trajectory.add_tensor(f'score', score)
        self.trajectory.add_value(f'sigma', sigma)
        self.trajectory.add_value(f'factor', factor)
        if epsilon is not None:
            self.trajectory.add_tensor(f'epsilon', epsilon)

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start


class LatentDiffusionSampler(DiffusionSampler):
    """
        Latent Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the latent diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__(scheduler, solver)

    def sample(self, model, z_start, SDE=False, record=False, verbose=False, return_latent=True):
        """
            Samples from the latent diffusion process using the specified model.

            Parameters:
                model (LatentDiffusionModel): Diffusion model supports 'score', 'tweedie', 'encode' and 'decode'
                z_start (torch.Tensor): Initial latent state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                return_latent (bool): Whether to return the latent state or decoded state.

            Returns:
                torch.Tensor: The final sampled state (latent or decoded).
        """
        if self.solver == 'euler':
            z0 = self._euler(model, z_start, SDE, record, verbose)
        else:
            raise NotImplementedError
        if return_latent:
            return z0
        else:
            x0 = model.decode(z0)
            return x0


class SPGD(nn.Module):
    """
        Langevin Dynamics sampling method.
    """

    def __init__(self, num_steps, lr, beta=0.9, lr_min_ratio=0.01):
        """
            Initializes the Langevin dynamics sampler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the sampling process.
                lr (float): Learning rate.
                lr_min_ratio (float): Minimum learning rate ratio.
        """
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.lr_min_ratio = lr_min_ratio
        self.beta = beta
        self.previous_grad = None


    def sample(self, xt, operator, measurement, sigma, factor, step_size, lr,
                model, record=False, verbose=False, **kwargs):
        """
            Samples using Langevin dynamics.

            Parameters:
                xt (torch.Tensor): Current state.
                operator (Operator): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                sigma (float): Current sigma value.
                factor (float): Current factor value.
                step_size (float): Current step size.
                lr (float): Current learning rate.
                model (nn.Module): Diffusion model.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                tuple: (x_t_minus_one, x0hat) The final sampled state and denoised estimate.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        
        x = xt.detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)

        for iter_step in pbar:
            # Tweedie estimation and score computation
            x0hat= model.tweedie(x, sigma)
            score = (x0hat - x) / sigma ** 2
            
            # Loss computation and optimization
            loss = operator.error(x0hat, measurement).sum()
            loss.backward(retain_graph=False)  # 使用 False 避免内存泄漏

            # adaptive momentum-based gradient smoothing.
            self._update_gradient(x.grad.detach(), iter_step)

            optimizer.step()
            optimizer.zero_grad()

            if torch.isnan(x).any():
                raise ValueError("NaN detected. Consider tuning (decreasing) learning rate.")

            # record
            if record:
                self._record(x, x0hat, loss)

        # when the searching is finished
        with torch.no_grad():
            x_t_minus_one = x + score * step_size * 0.5

        return x_t_minus_one.detach(), x0hat.detach()
    
    
    def _update_gradient(self, x_grad, iter_step):
        """adaptive momentum-based gradient smoothing."""
        if iter_step == 0:
            self.previous_grad = x_grad
        else:
            cos_sim = torch.cosine_similarity(x_grad, self.previous_grad, dim=1)
            alpha_j = (cos_sim.mean(dim=(1, 2)) + 1) / 2
            alpha_j = alpha_j.view(-1, 1, 1, 1)
            self.previous_grad = (alpha_j * self.beta) * self.previous_grad + (1 - alpha_j * self.beta) * x_grad

        return self.previous_grad

    def _record(self, x, x0hat, loss):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xi', x)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'loss', loss)

    def get_lr(self, ratio, p = 1):
        """
            Computes the learning rate based on the given ratio.
        """
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr


class RUN(nn.Module):
    """
        Implementation of decoupled annealing posterior sampling.
    """

    def __init__(self, annealing_scheduler_config, Tweedie_scheduler_config, config):
        """
            Initializes the required sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                Tweedie_scheduler_config (dict): Configuration for Tweedie estimate.
                config (dict): Configuration for SPGD algorithm.
        """
        super().__init__()
        annealing_scheduler_config, Tweedie_scheduler_config = self._check(annealing_scheduler_config,
                                                                             Tweedie_scheduler_config)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.Tweedie_scheduler_config = Tweedie_scheduler_config
        # Ensure config has required parameters
        if 'num_steps' not in config or 'lr' not in config:
            raise ValueError("SPGD config must contain 'num_steps' and 'lr' parameters")
        self.spgd = SPGD(**config)

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
            Samples using the SPGD method.

            Parameters:
                model (nn.Module): (Latent) Diffusion model
                x_start (torch.Tensor): Initial state.
                operator (nn.Module): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                evaluator (Evaluator): Evaluation function.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                **kwargs:
                    gt (torch.Tensor): reference ground truth data, only for evaluation

            Returns:
                torch.Tensor: The final sampled state.
        """
        if record:
            self.trajectory = Trajectory()

        pbar = tqdm.trange(self.annealing_scheduler.num_steps) if verbose else range(self.annealing_scheduler.num_steps)
        xt = x_start

        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]

            # Create Tweedie scheduler for current step
            Tweedie_scheduler = Scheduler(**self.Tweedie_scheduler_config, sigma_max=sigma)

            factor = Tweedie_scheduler.factor_steps[0]  # 2\dot\sigma(t)\sigma(t)\Delta t
            step_size = self.annealing_scheduler.factor_steps[step]
            lr = self.spgd.get_lr(step / self.annealing_scheduler.num_steps)
            # SPGD algorithm, searching the x_{t-1}
            xt, x0hat = self.spgd.sample(xt, operator, measurement, sigma, factor, step_size, lr, model, **kwargs)

            # Evaluation
            x0hat_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)

                # record
                if verbose and isinstance(pbar, tqdm.tqdm):
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0hat, sigma, x0hat_results)
        return xt

    def _record(self, xt, x0hat, sigma, x0hat_results):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])

    def _check(self, annealing_scheduler_config, Tweedie_scheduler_config):
        """
            Checks and updates the configurations for the schedulers.
        """
        # sigma_max of Tweedie scheduler change each step
        if 'sigma_max' in Tweedie_scheduler_config:
            Tweedie_scheduler_config.pop('sigma_max')

        # sigma final of annealing scheduler should always be 0
        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, Tweedie_scheduler_config

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
        return x_start


class Trajectory(nn.Module):
    """
        Class for recording and storing trajectory data.
    """

    def __init__(self):
        super().__init__()
        self.tensor_data = {}
        self.value_data = {}
        self._compile = False

    def add_tensor(self, name, images):
        """
            Adds image data to the trajectory.

            Parameters:
                name (str): Name of the image data.
                images (torch.Tensor): Image tensor to add.
        """
        if name not in self.tensor_data:
            self.tensor_data[name] = []
        self.tensor_data[name].append(images.detach().cpu())

    def add_value(self, name, values):
        """
            Adds value data to the trajectory.

            Parameters:
                name (str): Name of the value data.
                values (any): Value to add.
        """
        if name not in self.value_data:
            self.value_data[name] = []
        self.value_data[name].append(values)

    def compile(self):
        """
            Compiles the recorded data into tensors.

            Returns:
                Trajectory: The compiled trajectory object.
        """
        if not self._compile:
            self._compile = True
            for name in self.tensor_data.keys():
                self.tensor_data[name] = torch.stack(self.tensor_data[name], dim=0)
            for name in self.value_data.keys():
                self.value_data[name] = torch.tensor(self.value_data[name])
        return self

    @classmethod
    def merge(cls, trajs):
        """
            Merge a list of compiled trajectories from different batches

            Returns:
                Trajectory: The merged and compiled trajectory object.
        """
        merged_traj = cls()
        for name in trajs[0].tensor_data.keys():
            merged_traj.tensor_data[name] = torch.cat([traj.tensor_data[name] for traj in trajs], dim=1)
        for name in trajs[0].value_data.keys():
            merged_traj.value_data[name] = trajs[0].value_data[name]
        return merged_traj
