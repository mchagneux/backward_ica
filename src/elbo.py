from abc import ABCMeta, abstractmethod
from collections import namedtuple
from src.kalman import Kalman 
import torch 
from itertools import chain
import torch.nn as nn 
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


## Neural approximators for some ELBOs


class ApproximatorExpectationInQuadForm(nn.Module):
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        self.state_dim = state_dim 
        self.obs_dim = obs_dim
        input_shape = state_dim**2 + state_dim + state_dim**2
        self.A_tilde_approximator = nn.Sequential(nn.Linear(in_features=input_shape, out_features = obs_dim*state_dim, bias=True), nn.ReLU())
        self.a_tilde_approximator = nn.Sequential(nn.Linear(in_features=input_shape, out_features = obs_dim, bias=True), nn.ReLU())
        self.cov_tilde_approximator = nn.Sequential(nn.Linear(in_features=input_shape, out_features=(obs_dim * (obs_dim + 1)) // 2, bias=True), nn.ReLU())

    def forward(self, A, a, cov):
        inputs = torch.cat((A.flatten(), a, cov.flatten()))

        A_tilde = self.A_tilde_approximator(inputs).view((self.obs_dim, self.state_dim))
        a_tilde = self.a_tilde_approximator(inputs)

        cov_tilde_elements = self.cov_tilde_approximator(inputs)
        indices_to_fill = torch.triu_indices(self.obs_dim, self.obs_dim)
        cov_tilde = torch.empty((self.obs_dim, self.obs_dim))
        cov_tilde[indices_to_fill[0,:], indices_to_fill[1,:]] = cov_tilde_elements
        return A_tilde, a_tilde, cov_tilde @ cov_tilde.T


## ELBOs 


class BackwardELBO(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, p, q):
        super().__init__()

        self.p = p
        self.q = q

        self.dim_z = torch.as_tensor(self.p.transition.cov.shape[0])
        self.dim_x = torch.as_tensor(self.p.emission.cov.shape[0])

        self.backwards = [] 

        self.constants_V = None
        self.terms_V = None

    @abstractmethod
    def _update_V(self, observations):
        pass

    @abstractmethod
    def _init_V(self, observation):
        pass

    @abstractmethod
    def _expect_V_under_filtering():
        pass

    def update(self, observations):
        self._update_V(observations)
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


        self._init_V(observations[0])

        result =  self.update(observations[1:])
        self.backwards = []
        return result

class LinearGaussianQ(BackwardELBO, metaclass=ABCMeta):
        
    def __init__(self, p, q):
        super().__init__(p, q)

        self.kalman = Kalman(self.q)
        self.filtering = None
        # self.term_to_remove_if_V_update = None


    @abstractmethod
    def _expect_obs_term_under_backward(self, observation, backward_index):
        pass

    @abstractmethod
    def _expect_obs_term_under_filtering(self, observation):
        pass

    @abstractmethod
    def _get_obs_term(self, observation):
        pass

    def _expect_quad_form_under_backward(self, quad_form, backward_index=-1):
            constant = torch.trace(quad_form.Omega @ quad_form.A @ self.backwards[backward_index].cov @ quad_form.A.T)
            integrated_quad_form = QuadForm(Omega=quad_form.Omega, 
                                    A=quad_form.A @ self.backwards[backward_index].A, 
                                    b=quad_form.A @ self.backwards[backward_index].a + quad_form.b)
            return constant, integrated_quad_form
                            
    def _expect_quad_forms_under_backward(self, quad_forms, backward_index=-1):
        # expectation of (Au+b)^T Omega (Au+b) under the backward 
        constants = 0
        integrated_quad_forms = []
        for quad_form in quad_forms: 
            constant, integrated_quad_form = self._expect_quad_form_under_backward(quad_form, backward_index)
            constants += constant
            integrated_quad_forms.append(integrated_quad_form)

        return constants, integrated_quad_forms

    def _expect_transition_quad_form_under_backward(self, index=-1):
        # expectation of the quadratic form that appears in the log of the state transition density

        quad_form_in_z = QuadForm(Omega=-0.5*self.p_transition_prec, 
                                A=self.p.transition.map.weight @ self.backwards[index].A - torch.eye(self.p.transition.cov.shape[0]),
                                b=self.p.transition.map.weight @ self.backwards[index].a + self.p.transition.map.bias)
        return quad_form_in_z

    def _expect_quad_form_under_filtering(self, quad_form:QuadForm):
        return torch.trace(quad_form.Omega @ quad_form.A @ self.filtering.cov @ quad_form.A) + _eval_quad_form(quad_form, self.filtering.mean)

    def _update_backward(self):
        
        filtering_prec = torch.inverse(self.filtering.cov)

        backward_prec = self.q.transition.map.weight.T @ self.q_transition_prec @ self.q.transition.map.weight + filtering_prec

        backward_cov = torch.inverse(backward_prec)

        common_term = self.q.transition.map.weight.T @ self.q_transition_prec 
        backward_A = backward_cov @ common_term
        backward_a = backward_cov @ (filtering_prec @ self.filtering.mean - common_term @  self.q.transition.map.bias)

        self.backwards.append(Backward(A=backward_A, a=backward_a, cov=backward_cov))

    def _init_filtering(self, observation):
        self.filtering = Filtering(*self.kalman.init(observation)[2:])

    def _update_filtering(self, observation):
        self.filtering = Filtering(*self.kalman.filter_step(*self.filtering, 
                                            observation)[2:])

    def _init_V(self, observation):


        self.constants_V = torch.as_tensor(0., dtype=torch.float64)
        self.terms_V = []


        self._init_filtering(observation)

        self.constants_V += _constant_terms_from_log_gaussian(self.dim_z, torch.det(self.p.prior.cov)) + \
                            _constant_terms_from_log_gaussian(self.dim_x, self.p_emission_det_cov)
        

        quad_forms_to_add = [QuadForm(Omega=-0.5*torch.inverse(self.p.prior.cov), A=torch.eye(self.dim_z), b=-self.p.prior.mean), 
                            self._get_obs_term(observation)]
        self.terms_V.append(quad_forms_to_add)
        
    def _update_V(self, observations):

        # if self.term_to_remove_if_V_update is not None: self.constants_V -= self.term_to_remove_if_V_update

        for observation in torch.atleast_2d(observations):
            self._update_backward()
            self.constants_V += _constant_terms_from_log_gaussian(self.dim_x, self.p_emission_det_cov) \
                            +  _constant_terms_from_log_gaussian(self.dim_z, self.p_transition_det_cov) \
                            + -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.backwards[-1].cov)) \
                            +  0.5 * torch.as_tensor(self.dim_z, dtype=torch.float64) \
                            + - 0.5 * torch.trace(self.p_transition_prec @ self.p.transition.map.weight @ self.backwards[-1].cov @ self.p.transition.map.weight.T)
            
            quad_forms_to_add = [self._expect_transition_quad_form_under_backward(), self._get_obs_term(observation)]
            self.terms_V.append(quad_forms_to_add)
            self._update_filtering(observation)


        for backward_index, term in zip(range(len(self.backwards)), self.terms_V[:-1]):
            transition_term, obs_term = term
            constant, integrated_quad_form = self._expect_quad_form_under_backward(transition_term, backward_index=backward_index)
            self.constants_V += constant
            constant, integrated_obs_term = self._expect_obs_term_under_backward(obs_term, backward_index=backward_index)
            self.constants_V += constant 

            self.terms_V[backward_index] = [integrated_quad_form, integrated_obs_term]

        for quad_form_index, quad_forms in enumerate(self.terms_V):
            for backward_index in range(quad_form_index+1, len(self.backwards)):
                constants, quad_forms = self._expect_quad_forms_under_backward(quad_forms, backward_index=backward_index)
                self.constants_V += constants
            self.terms_V[quad_form_index] = quad_forms

        self.terms_V = list(chain(*self.terms_V))

        self.constants_V += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.filtering.cov))

    def _expect_V_under_filtering(self):
        
        result = self.constants_V 
        for term in self.terms_V[:-1]:
            result += self._expect_quad_form_under_filtering(term) 
        result += self._expect_obs_term_under_filtering(self.terms_V[-1]) # the last term is an observation term that has never been integrated: it is not a quad form in the nonlinear case
        result += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)

        return result

class LinearEmission(LinearGaussianQ):
    def __init__(self, p, q):
        super().__init__(p, q)

    def _expect_obs_term_under_backward(self, obs_term, backward_index):
        return self._expect_quad_form_under_backward(obs_term, backward_index)

    def _expect_obs_term_under_filtering(self, obs_term):
        return self._expect_quad_form_under_filtering(obs_term)

    def _get_obs_term(self, observation):
        return QuadForm(Omega=-0.5*self.p_emission_prec, 
                        A = self.p.emission.map.weight, 
                        b = self.p.emission.map.bias - observation)    

class NonLinearEmission(LinearGaussianQ):

        def __init__(self, p, q):
            super().__init__(p, q)
            self.approximator = ApproximatorExpectationInQuadForm(self.dim_z, self.dim_x)

        def _expect_obs_term_under_backward(self, obs_term, backward_index):
            observation = obs_term
            A_tilde, a_tilde, cov_tilde = self.approximator(*self.backwards[backward_index])
            constant_term = torch.trace(cov_tilde @ self.p_emission_prec)
            return constant_term, QuadForm(Omega=self.p_emission_prec, A=A_tilde, b=a_tilde - observation)

        def _expect_obs_term_under_filtering(self, obs_term):
            observation = obs_term
            _ , a_tilde, cov_tilde = self.approximator(A=torch.zeros((self.dim_x, self.dim_z)), a = self.filtering.mean, cov = self.filtering.cov)
            constant_term = torch.trace(cov_tilde @ self.p_emission_prec)
            return constant_term + (a_tilde - observation).T @ self.p_emission_prec @ (a_tilde - observation)

        def _get_obs_term(self, observation):
            return observation