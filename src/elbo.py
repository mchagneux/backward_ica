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


class BackwardELBO(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, model, v_model):
        super().__init__()

        self.model = model
        self.v_model = v_model

        self.dim_z = torch.as_tensor(self.model.transition.cov.shape[0])
        self.dim_x = torch.as_tensor(self.model.emission.cov.shape[0])

    
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

        self.model_transition_prec = torch.inverse(self.model.transition.cov)
        self.model_transition_det_cov = torch.det(self.model.transition.cov)
        self.model_emission_prec = torch.inverse(self.model.emission.cov)
        self.model_emission_det_cov = torch.det(self.model.emission.cov)


        self.v_model_transition_prec = torch.inverse(self.v_model.transition.cov)
        self.v_model_transition_det_cov = torch.det(self.v_model.transition.cov)
        self.v_model_emission_prec = torch.inverse(self.v_model.emission.cov)
        self.v_model_emission_det_cov = torch.det(self.v_model.emission.cov)


        self._init_V(observations[0])

        return self.update(observations[1:])


class QLinearGaussian(BackwardELBO):

    def __init__(self, model, v_model):
        super().__init__(model, v_model)

        self.kalman = Kalman(self.v_model)
        self.backwards = [] 
        self.filtering = None
        self.term_to_remove_if_V_update = None
        if isinstance(self.model.transition.map, nn.Linear): self._expect_obs_term_under_backward = self._expect_quad_form_under_backward            
        else:pass 


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

    def _expect_nonlinear_term_under_backward(self, term, backward_index):
        pass

    def _expect_transition_quad_form_under_backward(self, index=-1):
        # expectation of the quadratic form that appears in the log of the state transition density

        quad_form_in_z = QuadForm(Omega=-0.5*self.model_transition_prec, 
                                A=self.model.transition.map.weight @ self.backwards[index].A - torch.eye(self.model.transition.cov.shape[0]),
                                b=self.model.transition.map.weight @ self.backwards[index].a + self.model.transition.map.bias)
        return quad_form_in_z

    def _expect_quad_form_under_filtering(self, quad_form:QuadForm):
        return torch.trace(quad_form.Omega @ quad_form.A @ self.filtering.cov @ quad_form.A) + _eval_quad_form(quad_form, self.filtering.mean)

    def _update_backward(self):
        
        filtering_prec = torch.inverse(self.filtering.cov)

        backward_prec = self.v_model.transition.map.weight.T @ self.v_model_transition_prec @ self.v_model.transition.map.weight + filtering_prec

        backward_cov = torch.inverse(backward_prec)

        common_term = self.v_model.transition.map.weight.T @ self.v_model_transition_prec 
        backward_A = backward_cov @ common_term
        backward_a = backward_cov @ (filtering_prec @ self.filtering.mean - common_term @  self.v_model.transition.map.bias)

        self.backwards.append(Backward(A=backward_A, a=backward_a, cov=backward_cov))

    def _get_quad_form_in_z_obs_term(self, observation):
        return QuadForm(Omega=-0.5*self.model_emission_prec, 
                        A = self.model.emission.map.weight, 
                        b = self.model.emission.map.bias - observation)    

    def _init_filtering(self, observation):
        self.filtering = Filtering(*self.kalman.init(observation)[2:])

    def _update_filtering(self, observation):
        self.filtering = Filtering(*self.kalman.filter_step(*self.filtering, 
                                            observation)[2:])


    def _init_V(self, observation):


        self.constants_V = torch.as_tensor(0., dtype=torch.float64)
        self.terms_V = []


        self._init_filtering(observation)

        self.constants_V += _constant_terms_from_log_gaussian(self.dim_z, torch.det(self.model.prior.cov)) + \
                            _constant_terms_from_log_gaussian(self.dim_x, self.model_emission_det_cov)
        

        quad_forms_to_add = [self._get_quad_form_in_z_obs_term(observation), QuadForm(Omega=-0.5*torch.inverse(self.model.prior.cov), A=torch.eye(self.dim_z), b=-self.model.prior.mean)]
        self.terms_V.append(quad_forms_to_add)
        
    def _update_V(self, observations):

        # if self.term_to_remove_if_V_update is not None: self.constants_V -= self.term_to_remove_if_V_update


        for observation in torch.atleast_2d(observations):
            self._update_backward()
            self.constants_V += _constant_terms_from_log_gaussian(self.dim_x, self.model_emission_det_cov) \
                            +  _constant_terms_from_log_gaussian(self.dim_z, self.model_transition_det_cov) \
                            + -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.backwards[-1].cov)) \
                            +  0.5 * torch.as_tensor(self.dim_z, dtype=torch.float64) \
                            + - 0.5 * torch.trace(self.model_transition_prec @ self.model.transition.map.weight @ self.backwards[-1].cov @ self.model.transition.map.weight.T)
            
            quad_forms_to_add = [self._get_quad_form_in_z_obs_term(observation), self._expect_transition_quad_form_under_backward()]
            self.terms_V.append(quad_forms_to_add)
            self._update_filtering(observation)


        for backward_index, term in zip(range(len(self.backwards)), self.terms_V):
            obs_term, transition_term = term
            constant, integrated_obs_term = self._expect_obs_term_under_backward(obs_term, backward_index=backward_index)
            self.constants_V += constant 
            constant, integrated_quad_form = self._expect_quad_form_under_backward(transition_term, backward_index=backward_index)
            self.constants_V += constant
            self.terms_V[backward_index] = [integrated_obs_term, integrated_quad_form]

        for quad_form_index, quad_forms in enumerate(self.terms_V):
            for backward_index in range(quad_form_index+1, len(self.backwards)):
                constants, quad_forms = self._expect_quad_forms_under_backward(quad_forms, backward_index=backward_index)
                self.constants_V += constants
            self.terms_V[quad_form_index] = quad_forms

        self.terms_V = list(chain(*self.terms_V))

        self.constants_V += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.filtering.cov))

    def _expect_V_under_filtering(self):
        
        result = self.constants_V 
        for quad_form in self.terms_V:
            result += self._expect_quad_form_under_filtering(quad_form) 
        result += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)

        return result


    # def _old_update_V(self, observations):

        # for observation in torch.atleast_2d(observations): 
        #     self._update_backward()

        #     # dealing with all non-constant terms from previous V: one step before these quad forms were in z_next so they now are quad forms in z that need to be 
        #     # integrated against the backward, resulting in new quad forms in z_next
        #     for quad_form_nb, quad_form in enumerate(self.terms_V): 
        #         constant, integrated_quad_form = self._expect_quad_form_under_backward(quad_form)
        #         self.constants_V += constant
        #         self.terms_V[quad_form_nb] = integrated_quad_form


        #     # dealing with observation term seen as a quadratic form in z
        #     self.constants_V += _constant_terms_from_log_gaussian(self.dim_x, self.model_emission_det_cov)                                      
        #     self.terms_V.append(self._get_quad_form_in_z_obs_term(observation))

        #     # dealing with true transition term whose integration in z_previous under the backward is a quadratic form in z
        #     self.constants_V += _constant_terms_from_log_gaussian(self.dim_z, self.model_transition_det_cov)
        #     constant, integrated_quad_form = self._expect_transition_quad_form_under_backward()
        #     self.constants_V += constant 
        #     self.terms_V.append(integrated_quad_form)

        #     # dealing with backward term (integration of the quadratic form is just the dimension of z)
        #     self.constants_V += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.backward_cov))
        #     self.constants_V += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)


        #     self._update_filtering(observation)
        
        # # self.term_to_remove_if_V_update = -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.filtering_cov)).clone()
        # # self.constants_V += self.term_to_remove_if_V_update

        # self.constants_V += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.filtering_cov))
