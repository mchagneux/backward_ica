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
from functools import partial

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
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim
        self.v_approximator =  nn.Sequential(nn.Linear(in_features=target_dim, out_features=target_dim, bias=True), nn.SELU())
        self.W_approximator =  nn.Sequential(nn.Linear(in_features=target_dim, out_features=target_dim**2, bias=True), nn.SELU())


    def forward(self, filtering_params, backward_params, nonlinear_map):
        next_state_sample = MultivariateNormal(*filtering_params).sample()
        backward_sample = MultivariateNormal(loc=backward_params.A @ next_state_sample + backward_params.a, 
                                            covariance_matrix=backward_params.cov).sample() 
        mapped_sample = nonlinear_map(backward_sample)
        return self.v_approximator(mapped_sample), self.W_approximator(mapped_sample).view((self.target_dim, self.target_dim))







## ELBOs 


class BackwardELBO(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, p, q):
        super().__init__()

        self.p = p
        self.q = q

        self.dim_z = torch.as_tensor(self.p.transition.cov.shape[0])
        self.dim_x = torch.as_tensor(self.p.emission.cov.shape[0])

        self.constants_V = torch.as_tensor(0., dtype=torch.float64)

        self.backward_params = Backward(A=torch.empty(size=(self.dim_z, self.dim_z)), 
                                a=torch.empty(size=(self.dim_z,)), 
                                cov=torch.empty(size=(self.dim_z, self.dim_z)))

        self.filtering_params = Filtering(mean=torch.empty(size=(self.dim_z,)),
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
        self._update_filtering(observation)
        self._update_V(observation)

    def update(self, observation):
        self.constants_V -= -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.filtering_params.cov))
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
        self.constants_V += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.filtering_params.cov))
        
        result =  self._expect_V_under_filtering()

        ## temporary workaround 
        self.__init__(self.p, self.q)
        return result

class LinearGaussianQ(BackwardELBO, metaclass=ABCMeta):
        
    def __init__(self, p, q):
        super().__init__(p, q)

        self.kalman = Kalman(self.q)
        self.obs_terms = []
        self.transition_terms = []


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
        

        self.transition_terms.append(QuadForm(Omega=-0.5*torch.inverse(self.p.prior.cov), A=torch.eye(self.dim_z), b=-self.p.prior.mean))
        self.obs_terms.append(self._get_obs_term(observation))

    def _integrate_previous_terms(self):
        for term_nb, (transition_term, obs_term) in enumerate(zip(self.transition_terms, self.obs_terms)):
            constants_0, self.transition_terms[term_nb] = self._expect_quad_form_under_backward(transition_term)
            constants_1, self.obs_terms[term_nb] = self._expect_obs_term_under_backward(obs_term)
            self.constants_V += constants_0 + constants_1

    def _update_V(self, observation):


        self._integrate_previous_terms()

        self.constants_V += _constant_terms_from_log_gaussian(self.dim_x, self.p_emission_det_cov) \
                        +  _constant_terms_from_log_gaussian(self.dim_z, self.p_transition_det_cov) \
                        + -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.backward_params.cov)) \
                        +  0.5 * torch.as_tensor(self.dim_z, dtype=torch.float64) \
                        + - 0.5 * torch.trace(self.p_transition_prec @ self.p.transition.map.weight @ self.backward_params.cov @ self.p.transition.map.weight.T)
        
        self.transition_terms.append(self._expect_transition_quad_form_under_backward())
        self.obs_terms.append(self._get_obs_term(observation))

    def _expect_V_under_filtering(self):

        result = self.constants_V 

        for transition_term, obs_term in zip(self.transition_terms[:-1], self.obs_terms[:-1]):
            result += self._expect_quad_form_under_filtering(transition_term) \
                    + self._expect_quad_form_under_filtering(obs_term)


        result += self._expect_quad_form_under_filtering(self.transition_terms[-1]) 
        result += self._expect_obs_term_under_filtering(self.obs_terms[-1]) 

        result += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)

        return result

    def _init_filtering(self, observation):
        self.filtering_params = Filtering(*self.kalman.init(observation)[2:])
    
    def _update_filtering(self, observation):
        self.filtering_params = Filtering(*self.kalman.filter_step(*self.filtering_params, observation)[2:])

    def _update_backward(self):
        
        filtering_prec = torch.inverse(self.filtering_params.cov)

        backward_prec = self.q.transition.map.weight.T @ self.q_transition_prec @ self.q.transition.map.weight + filtering_prec

        backward_cov = torch.inverse(backward_prec)

        common_term = self.q.transition.map.weight.T @ self.q_transition_prec 
        backward_A = backward_cov @ common_term
        backward_a = backward_cov @ (filtering_prec @ self.filtering_params.mean - common_term @  self.q.transition.map.bias)

        self.backward_params = Backward(A=backward_A, a=backward_a, cov=backward_cov)

    def _expect_quad_form_under_backward(self, quad_form):
        constant = torch.trace(quad_form.Omega @ quad_form.A @ self.backward_params.cov @ quad_form.A.T)
        integrated_quad_form = QuadForm(Omega=quad_form.Omega, 
                                A=quad_form.A @ self.backward_params.A, 
                                b=quad_form.A @ self.backward_params.a + quad_form.b)
        return constant, integrated_quad_form
                        
    def _expect_transition_quad_form_under_backward(self):
        # expectation of the quadratic form that appears in the log of the state transition density

        quad_form_in_z = QuadForm(Omega=-0.5*self.p_transition_prec, 
                                A=self.p.transition.map.weight @ self.backward_params.A - torch.eye(self.p.transition.cov.shape[0]),
                                b=self.p.transition.map.weight @ self.backward_params.a + self.p.transition.map.bias)
        return quad_form_in_z

    def _expect_quad_form_under_filtering(self, quad_form:QuadForm):
        return torch.trace(quad_form.Omega @ quad_form.A @ self.filtering_params.cov @ quad_form.A) + _eval_quad_form(quad_form, self.filtering_params.mean)

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
            self.approximator = QuadFormParametersFromExpectationOfNonLinearTerm(target_dim=self.dim_x)
            
        def _expect_obs_term_under_backward(self, obs_term):
            observation = obs_term
            nonlinear_map = partial(_quad_form_in_emission_term(observation=observation, 
                                                                emission_map=self.p.emission.map, 
                                                                emission_prec=self.p_emission_prec))
            
            v, W = self.approximator(filtering_params=self.filtering_params, 
                                    backward_params=self.backward_params, 
                                    nonlinear_map=nonlinear_map)

            return QuadForm(Omega=W, 
                            A=torch.eye(self.dim_x),
                            b=-v)
                        
        def _expect_obs_term_under_filtering(self, obs_term):
            observation = obs_term
            _ , a_tilde, cov_tilde = self.approximator(A=torch.zeros((self.dim_x, self.dim_z)), a = self.filtering_params.mean, cov = self.filtering_params.cov)
            constant_term = torch.trace(cov_tilde @ self.p_emission_prec)
            return constant_term + (a_tilde - observation).T @ self.p_emission_prec @ (a_tilde - observation)

        def _get_obs_term(self, observation):
            return observation





