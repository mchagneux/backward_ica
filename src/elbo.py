from collections import namedtuple
from src.kalman import Kalman 
import torch 
import torch.nn as nn 
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

pi = torch.as_tensor(torch.pi)

QuadForm = namedtuple('QuadForm',['Omega','A','b'])

def _constant_terms_from_log_gaussian(dim, det_cov):
            return -0.5*(dim* torch.log(2*pi) + torch.log(det_cov))

def _eval_quad_form(quad_form, x):
        common_term = quad_form.A @ x + quad_form.b
        return common_term.T @ quad_form.Omega @ common_term

class LinearGaussianELBO(torch.nn.Module):

    def __init__(self, model:nn.ParameterDict, v_model):
        super().__init__()

        self.model = model
        self.v_model = v_model

        self.dim_z = torch.as_tensor(self.model.transition.cov.shape[0])
        self.dim_x = torch.as_tensor(self.model.emission.cov.shape[0])

        
        self.kalman = Kalman(self.v_model)

        self.backward_cov = None 
        self.backward_A = None 
        self.backward_a = None 
        self.filtering_mean = None 
        self.filtering_cov = None


        self.constants_V = None
        self.quad_forms_V = None
        self.term_to_remove_if_V_update = None



    def _expect_quad_form_under_backward(self, quad_form:QuadForm):
        # expectation of (Au+b)^T Omega (Au+b) under the backward 

        constants = torch.trace(quad_form.Omega @ quad_form.A @ self.backward_cov @ quad_form.A.T)
        quad_form_in_z = QuadForm(Omega=quad_form.Omega, 
                                A=quad_form.A @ self.backward_A, 
                                b=quad_form.A @ self.backward_a + quad_form.b)
        return constants, quad_form_in_z

    def _expect_transition_quad_form_under_backward(self):
        # expectation of the quadratic form that appears in the log of the state transition density

        constants =  - 0.5 * torch.trace(self.model_transition_prec @ self.model.transition.map.weight @ self.backward_cov @ self.model.transition.map.weight.T)
        quad_form_in_z = QuadForm(Omega=-0.5*self.model_transition_prec, 
                                A=self.model.transition.map.weight @ self.backward_A - torch.eye(self.model.transition.cov.shape[0]),
                                b=self.model.transition.map.weight @ self.backward_a + self.model.transition.map.bias)
        return constants, quad_form_in_z

    def _expect_quad_form_under_filtering(self, quad_form:QuadForm):
        return torch.trace(quad_form.Omega @ quad_form.A @ self.filtering_cov @ quad_form.A) + _eval_quad_form(quad_form, self.filtering_mean)

    def _update_backward(self):
        
        filtering_prec = torch.inverse(self.filtering_cov)

        backward_prec = self.v_model.transition.map.weight.T @ self.v_model_transition_prec @ self.v_model.transition.map.weight + filtering_prec

        self.backward_cov = torch.inverse(backward_prec)

        common_term = self.v_model.transition.map.weight.T @ self.v_model_transition_prec 
        self.backward_A = self.backward_cov @ common_term
        self.backward_a = self.backward_cov @ (filtering_prec @ self.filtering_mean - common_term @  self.v_model.transition.map.bias)

    def _get_quad_form_in_z_obs_term(self, observation):
        return QuadForm(Omega=-0.5*self.model_emission_prec, 
                        A = self.model.emission.map.weight, 
                        b = self.model.emission.map.bias - observation)    

    def _init_filtering(self, observation):
        self.filtering_mean, self.filtering_cov = self.kalman.init(observation)[2:]

    def _update_filtering(self, observation):
        self.filtering_mean, self.filtering_cov = self.kalman.filter_step(self.filtering_mean, 
                                            self.filtering_cov, 
                                            observation)[2:]
                
    def init_V(self, observation):

        self._init_filtering(observation)

        self.constants_V = torch.as_tensor(0., dtype=torch.float64)
        self.quad_forms_V = []

        self.constants_V += _constant_terms_from_log_gaussian(self.dim_z, torch.det(self.model.prior.cov)) + \
                    _constant_terms_from_log_gaussian(self.dim_x, self.model_emission_det_cov)
        

        self.quad_forms_V.append(self._get_quad_form_in_z_obs_term(observation))
        self.quad_forms_V.append(QuadForm(Omega=-0.5*torch.inverse(self.model.prior.cov), A=torch.eye(self.dim_z), b=-self.model.prior.mean))
        
    def update_V(self, observations):

        if self.term_to_remove_if_V_update is not None: self.constants_V -= self.term_to_remove_if_V_update

        for observation in observations: 
            self._update_backward()

            # dealing with all non-constant terms from previous V: one step before these quad forms were in z_next so they now are quad forms in z that need to be 
            # integrated against the backward, resulting in new quad forms in z_next
            for quad_form_nb, quad_form in enumerate(self.quad_forms_V): 
                constant, integrated_quad_form = self._expect_quad_form_under_backward(quad_form)
                self.constants_V += constant
                self.quad_forms_V[quad_form_nb] = integrated_quad_form


            # dealing with observation term seen as a quadratic form in z
            self.constants_V += _constant_terms_from_log_gaussian(self.dim_x, self.model_emission_det_cov)                                      
            self.quad_forms_V.append(self._get_quad_form_in_z_obs_term(observation))

            # dealing with true transition term whose integration in z_previous under the backward is a quadratic form in z
            self.constants_V += _constant_terms_from_log_gaussian(self.dim_z, self.model_transition_det_cov)
            constant, integrated_quad_form = self._expect_transition_quad_form_under_backward()
            self.constants_V += constant 
            self.quad_forms_V.append(integrated_quad_form)

            # dealing with backward term (integration of the quadratic form is just the dimension of z)
            self.constants_V += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.backward_cov))
            self.constants_V += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)


            self._update_filtering(observation)
        
        self.term_to_remove_if_V_update = -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.filtering_cov)).clone()
        self.constants_V += self.term_to_remove_if_V_update

    def _expect_V_under_filtering(self):
        
        result = self.constants_V 
        for quad_form in self.quad_forms_V:
            result += self._expect_quad_form_under_filtering(quad_form) 
        result += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)

        return result

    def update(self, observations):
        self.update_V(observations)
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


        self.init_V(observations[0])

        return self.update(observations[1:])





        



        
        


