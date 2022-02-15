from collections import namedtuple
from utils.kalman import Kalman 
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
        
        self.kalman = Kalman(self.v_model)

        self.backward_cov = None 
        self.backward_A = None 
        self.backward_a = None 
        self.filtering_mean = None 
        self.filtering_cov = None

        self.dim_z = torch.as_tensor(self.model.transition.cov.shape[0])
        self.dim_x = torch.as_tensor(self.model.emission.cov.shape[0])

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

    def _expectations_under_backward(self, quad_forms, observation):

        new_constants = 0
        new_quad_forms = []

        # dealing with all non-constant terms from previous V: one step before these quad forms were in z_next so they now are quad forms in z that need to be 
        # integrated against the backward, resulting in new quad forms in z_next
        for quad_form in quad_forms: 
            constant, integrated_quad_form = self._expect_quad_form_under_backward(quad_form)
            new_constants += constant
            new_quad_forms.append(integrated_quad_form)


        # dealing with observation term seen as a quadratic form in z
        new_constants += _constant_terms_from_log_gaussian(self.dim_x, self.model_emission_det_cov)                                      
        new_quad_forms.append(self._get_quad_form_in_z_obs_term(observation))

        # dealing with true transition term whose integration in z_previous under the backward is a quadratic form in z
        new_constants += _constant_terms_from_log_gaussian(self.dim_z, self.model_transition_det_cov)
        constant, integrated_quad_form = self._expect_transition_quad_form_under_backward()
        new_constants += constant 
        new_quad_forms.append(integrated_quad_form)

        # dealing with backward term (integration of the quadratic form is just the dimension of z)
        new_constants += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.backward_cov))
        new_constants += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)
        
        
        return new_constants, new_quad_forms

    def _init_filtering(self, observation):
        self.filtering_mean, self.filtering_cov = self.kalman.init(observation)[2:]

    def _update_filtering(self, observation):
        self.filtering_mean, self.filtering_cov = self.kalman.filter_step(self.filtering_mean, 
                                            self.filtering_cov, 
                                            observation)[2:]
                
    def forward(self, observations):

        self.model_transition_prec = torch.inverse(self.model.transition.cov)
        self.model_transition_det_cov = torch.det(self.model.transition.cov)
        self.model_emission_prec = torch.inverse(self.model.emission.cov)
        self.model_emission_det_cov = torch.det(self.model.emission.cov)


        self.v_model_transition_prec = torch.inverse(self.v_model.transition.cov)
        self.v_model_transition_det_cov = torch.det(self.v_model.transition.cov)
        self.v_model_emission_prec = torch.inverse(self.v_model.emission.cov)
        self.v_model_emission_det_cov = torch.det(self.v_model.emission.cov)

        self._init_filtering(observations[0])

        constants = torch.as_tensor(0., dtype=torch.float64)
        quad_forms = []

        constants += _constant_terms_from_log_gaussian(self.dim_z, torch.det(self.model.prior.cov)) + \
                    _constant_terms_from_log_gaussian(self.dim_x, self.model_emission_det_cov)
        

        quad_forms.append(self._get_quad_form_in_z_obs_term(observations[0]))
        quad_forms.append(QuadForm(Omega=-0.5*torch.inverse(self.model.prior.cov), A=torch.eye(self.dim_z), b=-self.model.prior.mean))


        for observation in observations[1:]:

            self._update_backward()

            new_constants, new_quad_forms = self._expectations_under_backward(quad_forms, observation)


            constants += new_constants
            quad_forms = new_quad_forms

            self._update_filtering(observation)


        constants += -_constant_terms_from_log_gaussian(self.dim_z, torch.det(self.filtering_cov))

        for quad_form in quad_forms:
            constants += self._expect_quad_form_under_filtering(quad_form) 
        constants += 0.5*torch.as_tensor(self.dim_z, dtype=torch.float64)

        return constants 







        



        
        


