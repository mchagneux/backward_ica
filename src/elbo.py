from abc import ABCMeta, abstractmethod
from collections import namedtuple
from tkinter import N
from src.kalman import Kalman 
import torch 
from itertools import chain
import torch.nn as nn 
from torch.distributions import MultivariateNormal
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

pi = torch.as_tensor(torch.pi)

QuadForm = namedtuple('QuadForm',['Omega','A','b'])
Backward = namedtuple('Backward',('A','a','cov'))
Filtering = namedtuple('Filtering',('mean','cov'))

def _constant_terms_from_log_gaussian(dim, det_cov):
    
    return -0.5*(dim* torch.log(2*pi) + torch.log(det_cov))

def _eval_quad_form(quad_form, x):
    common_term = quad_form.A @ x + quad_form.b
    return common_term.T @ quad_form.Omega @ common_term

def get_appropriate_elbo(q_description, p_description):
    if q_description == 'linear_gaussian':
        if p_description == 'linear_emission': return LinearEmission
        elif p_description == 'nonlinear_emission': return NonLinearEmission
        else: raise NotImplementedError
    else: 
        raise NotImplementedError

def _quad_form_in_emission_term(observation, emission_map, emission_prec, z):
    common_term = observation - emission_map(z)
    return common_term.T @ emission_prec @ common_term

## Neural approximators for some ELBOs
class QuadFormParametersFromExpectationOfNonLinearTerm(nn.Module):
    def __init__(self, hidden_dim, target_dim):
        super().__init__()
        self.target_dim = target_dim
        self.v_approximator =  nn.Sequential(nn.Linear(in_features=1, out_features=64, bias=True), nn.SELU(),
                                            nn.Linear(in_features=64, out_features=target_dim, bias=True), nn.SELU())

        self.W_approximator =  nn.Sequential(nn.Linear(in_features=1, out_features=64, bias=True), nn.SELU(),
                                            nn.Linear(in_features=64, out_features=target_dim**2, bias=True), nn.SELU())

    def forward(self, q_predictive, q_backward, nonlinear_map):
        next_state_sample = MultivariateNormal(*q_predictive).sample()
        backward_sample = MultivariateNormal(loc=q_backward.A @ next_state_sample + q_backward.a, 
                                            covariance_matrix=q_backward.cov).sample() 
        mapped_sample = torch.atleast_1d(nonlinear_map(backward_sample))
        v = self.v_approximator(mapped_sample)
        W = self.W_approximator(mapped_sample).view((self.target_dim, self.target_dim))

        return v, W



## ELBOs 
class BackwardELBO(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, p, q):
        super().__init__()

        self.p = p
        self.q = q

        self.dim_z = torch.as_tensor(self.p.transition.cov.shape[0])
        self.dim_x = torch.as_tensor(self.p.emission.cov.shape[0])

        self.constants_V = torch.as_tensor(0., dtype=torch.float64)

        self.q_backward = Backward(A=torch.empty(size=(self.dim_z, self.dim_z)), 
                                a=torch.empty(size=(self.dim_z,)), 
                                cov=torch.empty(size=(self.dim_z, self.dim_z)))

        self.q_filtering = Filtering(mean=torch.empty(size=(self.dim_z,)),
                                cov=torch.empty(size=(self.dim_z,self.dim_z)))

    @abstractmethod
    def _init_filtering(self, observation):
        pass

    @abstractmethod
    def _update_filtering(self, observation):
        pass

    @abstractmethod
    def _update_backward(self):
        pass 

    @abstractmethod
    def _init_V(self, observation):
        pass

    @abstractmethod
    def _update_V(self, observation):
        pass

    @abstractmethod
    def _expect_V_under_filtering():
        pass

    def init(self, observation):
        self._init_V(observation)
        self._init_filtering(observation)

    def _update(self, observation):
        self._update_backward()
        self._update_V(observation)
        self._update_filtering(observation)


    def update(self, observation):
        self.constants_V -= -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.q_filtering.cov))
        self._update(observation)
        return self._expect_V_under_filtering()

    def forward(self, observations):

        self.p_transition_prec = torch.inverse(self.p.transition.cov)
        self.p_transition_det_cov = torch.det(self.p.transition.cov)
        self.p_emission_prec = torch.inverse(self.p.emission.cov)
        self.p_emission_det_cov = torch.det(self.p.emission.cov)


        self.q_transition_prec = torch.inverse(self.q.transition.cov)
        self.q_transition_det_cov = torch.det(self.q.transition.cov)
        self.q_emission_prec = torch.inverse(self.q.emission.cov)
        self.q_emission_det_cov = torch.det(self.q.emission.cov)

        self.init(observations[0])

        for observation in observations[1:]:
            self._update(observation)
        self.constants_V += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.q_filtering.cov))
        
        result =  self._expect_V_under_filtering()

        self.__init__(self.p, self.q)        ## temporary workaround 

        return result

class LinearGaussianQ(BackwardELBO, metaclass=ABCMeta):
        
    def __init__(self, p, q):
        super().__init__(p, q)

        self.kalman = Kalman(self.q)
        self.nonlinear_terms = []
        self.quad_forms = []


    @abstractmethod
    def _expect_obs_term_under_backward(self, obs_term):
        pass

    @abstractmethod
    def _expect_obs_term_under_filtering(self, observation):
        pass

    @abstractmethod
    def _get_obs_term(self, observation):
        pass

    def _init_V(self, observation):

        self.constants_V += _constant_terms_from_log_gaussian(self.dim_z, torch.det(self.p.prior.cov)) + \
                            _constant_terms_from_log_gaussian(self.dim_x, self.p_emission_det_cov)
        

        self.quad_forms.append(QuadForm(Omega=-0.5*torch.inverse(self.p.prior.cov), A=torch.eye(self.dim_z), b=-self.p.prior.mean))
        self.nonlinear_terms.append(self._get_obs_term(observation))

    def _integrate_previous_terms(self):
        for term_nb, quad_form in enumerate(self.quad_forms):
            constants, self.quad_forms[term_nb] = self._expect_quad_form_under_backward(quad_form)
            self.constants_V += constants
        for term_nb in range(len(self.nonlinear_terms)):
            constants, integrated_nonlinear_term = self._expect_obs_term_under_backward(self.nonlinear_terms.pop(term_nb))
            self.quad_forms.append(integrated_nonlinear_term)
            self.constants_V += constants


    def _update_V(self, observation):

        self._integrate_previous_terms()

        self.constants_V += _constant_terms_from_log_gaussian(self.dim_x, self.p_emission_det_cov) \
                        +  _constant_terms_from_log_gaussian(self.dim_z, self.p_transition_det_cov) \
                        + -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.q_backward.cov)) \
                        +  0.5 * torch.as_tensor(self.dim_z, dtype=torch.float64) \
                        + - 0.5 * torch.trace(self.p_transition_prec @ self.p.transition.map.weight @ self.q_backward.cov @ self.p.transition.map.weight.T)
        
        self.quad_forms.append(self._expect_transition_quad_form_under_backward())
        self.nonlinear_terms.append(self._get_obs_term(observation))

    def _expect_V_under_filtering(self):

        result = self.constants_V 

        for quad_form in self.quad_forms:
            result += self._expect_quad_form_under_filtering(quad_form)

        for nonlinear_term in self.nonlinear_terms:
            result += self._expect_obs_term_under_filtering(nonlinear_term)

        result += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)

        return result

    def _init_filtering(self, observation):
        self.q_filtering = Filtering(*self.kalman.init(observation)[2:])
    
    def _update_filtering(self, observation):
        self.q_filtering = Filtering(*self.kalman.filter_step(*self.q_filtering, observation)[2:])

    def _update_backward(self):
        
        filtering_prec = torch.inverse(self.q_filtering.cov)

        backward_prec = self.q.transition.map.weight.T @ self.q_transition_prec @ self.q.transition.map.weight + filtering_prec

        backward_cov = torch.inverse(backward_prec)

        common_term = self.q.transition.map.weight.T @ self.q_transition_prec 
        backward_A = backward_cov @ common_term
        backward_a = backward_cov @ (filtering_prec @ self.q_filtering.mean - common_term @  self.q.transition.map.bias)

        self.q_backward = Backward(A=backward_A, a=backward_a, cov=backward_cov)

    def _expect_quad_form_under_backward(self, quad_form):
        constant = torch.trace(quad_form.Omega @ quad_form.A @ self.q_backward.cov @ quad_form.A.T)
        integrated_quad_form = QuadForm(Omega=quad_form.Omega, 
                                A=quad_form.A @ self.q_backward.A, 
                                b=quad_form.A @ self.q_backward.a + quad_form.b)
        return constant, integrated_quad_form
                        
    def _expect_transition_quad_form_under_backward(self):
        # expectation of the quadratic form that appears in the log of the state transition density

        quad_form_in_z = QuadForm(Omega=-0.5*self.p_transition_prec, 
                                A=self.p.transition.map.weight @ self.q_backward.A - torch.eye(self.p.transition.cov.shape[0]),
                                b=self.p.transition.map.weight @ self.q_backward.a + self.p.transition.map.bias)
        return quad_form_in_z

    def _expect_quad_form_under_filtering(self, quad_form:QuadForm):
        return torch.trace(quad_form.Omega @ quad_form.A @ self.q_filtering.cov @ quad_form.A) + _eval_quad_form(quad_form, self.q_filtering.mean)

class LinearEmission(LinearGaussianQ):
    def __init__(self, p, q):
        super().__init__(p, q)

    def _expect_obs_term_under_backward(self, obs_term):
        return self._expect_quad_form_under_backward(obs_term)

    def _expect_obs_term_under_filtering(self, obs_term):
        return self._expect_quad_form_under_filtering(obs_term)

    def _get_obs_term(self, observation):
        return QuadForm(Omega=-0.5*self.p_emission_prec, 
                        A = self.p.emission.map.weight, 
                        b = self.p.emission.map.bias - observation)    

class NonLinearEmission(LinearGaussianQ):

        def __init__(self, p, q):
            super().__init__(p, q)
            self.approximator = QuadFormParametersFromExpectationOfNonLinearTerm(hidden_dim=128, target_dim=self.dim_z)
            
        def _expect_obs_term_under_backward(self, obs_term):
            
            v, W = self.approximator(q_predictive=self.kalman.predict(*self.q_filtering), 
                                    q_backward=self.q_backward, 
                                    nonlinear_map=obs_term)

            return 0.0, QuadForm(Omega=W, 
                            A=torch.eye(self.dim_z),
                            b=-v)
                        
        def _expect_obs_term_under_filtering(self, obs_term):
            return obs_term(MultivariateNormal(loc=self.q_filtering.mean, 
                                                    covariance_matrix=self.q_filtering.cov).sample())

        def _get_obs_term(self, observation):
            return lambda z: _quad_form_in_emission_term(observation=observation, 
                                                                emission_map=self.p.emission.map,
                                                                emission_prec=self.p_emission_prec, 
                                                                z=z)





