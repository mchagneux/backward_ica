from abc import ABC, abstractmethod
from logging import Filter
from turtle import back
from typing import Callable
from utils.kalman import Kalman
from utils.misc import inv, QuadForm, constant_terms_from_log_gaussian, Id
from utils.linear_gaussian_hmm import HMM, LinearGaussianHMM
import numpy as np
from typing import Collection



class ELBO(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def compute(self, observations):
        pass 


# specifying computations of V for the case where both true model and the variational model are Linear Gaussian HMMs
class ELBOLinearGaussian(ELBO):
    
    def __init__(self, 
                true_model: LinearGaussianHMM, 
                variational_model: LinearGaussianHMM):


        self.kalman = Kalman(model=variational_model)

        self.A_theta = true_model.state_transition_matrix
        self.a_theta = true_model.state_transition_offset
        self.Q_theta = true_model.state_covariance_matrix
        self.inv_Q_theta = inv(self.Q_theta)
        self.det_Q_theta = np.linalg.det(self.Q_theta)

        self.B_theta = true_model.observation_matrix
        self.b_theta = true_model.observation_offset
        self.R_theta = true_model.observation_covariance_matrix
        self.inv_R_theta = inv(self.R_theta)
        self.det_R_theta = np.linalg.det(self.R_theta)

        self.prior_mean_theta, self.prior_cov_theta = true_model.prior_mean, true_model.prior_cov
        self.prior_mean_phi, self.prior_cov_phi = variational_model.prior_mean, variational_model.prior_cov

        self.A_phi = variational_model.state_transition_matrix
        self.a_phi = variational_model.state_transition_offset
        self.Q_phi = variational_model.state_covariance_matrix
        self.inv_Q_phi  = inv(self.Q_phi)
        self.det_Q_phi = np.linalg.det(self.Q_phi)

        self.dim_z = len(self.a_theta)
        self.dim_x = len(self.b_theta)

    def _expect_quad_form_in_z_under_backward(self, quad_form:QuadForm, A_backward, a_backward, backward_cov):
        # expectation of (Au+b)^T Omega (Au+b) under the backward 

        constants = np.trace(quad_form.Omega @ quad_form.A @ backward_cov @ quad_form.A.T)
        quad_form_in_z = QuadForm(Omega=quad_form.Omega, 
                                A=quad_form.A @ A_backward, 
                                b=quad_form.A @ a_backward + quad_form.b)
        return constants, quad_form_in_z

    def _expect_quad_form_state_transition_under_backward(self, A_backward, a_backward, backward_cov):
        # expectation of the quadratic form that appears in the log of the state transition density

        constants =  - 0.5 * np.trace(self.inv_Q_theta @ self.A_theta @ backward_cov @ self.A_theta.T)
        quad_form_in_z = QuadForm(Omega=-0.5*self.inv_Q_theta, 
                                A=self.A_theta @ A_backward - Id(self.dim_z),
                                b=self.A_theta @ a_backward + self.a_theta)
        return constants, quad_form_in_z

    def _expect_quad_form_under_filtering(self, quad_form:QuadForm, filtering_mean, filtering_cov):
        return np.trace(quad_form.Omega @ quad_form.A @ filtering_cov @ quad_form.A.T) + quad_form(filtering_mean)

    def _init_filtering(self, observation):
        _, _, filtering_mean, filtering_cov = self.kalman.init(observation)
        return filtering_mean, filtering_cov

    def _update_filtering(self, observation, filtering_mean, filtering_cov):
        _, _, filtering_mean, filtering_cov = self.kalman.filter_step(filtering_mean, filtering_cov, observation)
        return filtering_mean, filtering_cov

    def _update_backward(self, filtering_mean, filtering_cov):

        filtering_prec = inv(filtering_cov)

        backward_prec = self.A_phi.T @ self.inv_Q_phi @ self.A_phi + filtering_prec

        backward_cov = inv(backward_prec)

        common_term = self.A_phi.T @ self.inv_Q_phi 
        A_backward = backward_cov @ common_term
        a_backward = backward_cov @ (filtering_prec @ filtering_mean - common_term @ self.a_phi)

        return A_backward, a_backward, backward_cov

    def _get_quad_form_in_z_obs_term(self, observation):
        return QuadForm(Omega=-0.5*self.inv_R_theta, 
                        A = self.B_theta, 
                        b = self.b_theta - observation)    

    def _expectations_under_backward(self, quad_forms_in_z, A_backward, a_backward, backward_cov, observation):

        new_constants = 0
        new_quad_forms_in_z = []

        # dealing with all non-constant terms from previous V: one step before these quad forms were in z_next so they now are quad forms in z that need to be 
        # integrated against the backward, resulting in new quad forms in z_next
        for quad_form in quad_forms_in_z: 
            constant, integrated_quad_form = self._expect_quad_form_in_z_under_backward(quad_form, A_backward, a_backward, backward_cov)
            new_constants += constant
            new_quad_forms_in_z.append(integrated_quad_form)


        # dealing with observation term seen as a quadratic form in z
        new_constants += constant_terms_from_log_gaussian(self.dim_x, self.det_R_theta)                                      
        new_quad_forms_in_z.append(self._get_quad_form_in_z_obs_term(observation))

        # dealing with true transition term whose integration in z_previous under the backward is a quadratic form in z
        new_constants += constant_terms_from_log_gaussian(self.dim_z, self.det_Q_theta)
        constant, integrated_quad_form = self._expect_quad_form_state_transition_under_backward(A_backward, a_backward, backward_cov)
        new_constants += constant 
        new_quad_forms_in_z.append(integrated_quad_form)

        # dealing with backward term (integration of the quadratic form is just the dimension of z)
        new_constants += -constant_terms_from_log_gaussian(self.dim_z, np.linalg.det(backward_cov))
        new_constants += 0.5*self.dim_z

        
        return new_constants, new_quad_forms_in_z

    def _expectations_under_filtering(self, quad_forms_in_z, filtering_mean, filtering_cov):
        new_constants = 0

        for quad_form in quad_forms_in_z:
            new_constants += self._expect_quad_form_under_filtering(quad_form, filtering_mean, filtering_cov)
        
        return new_constants

    def init_V(self, observation):
        init_constants = 0 
        init_quad_forms_in_z = []

        init_constants += constant_terms_from_log_gaussian(self.dim_z, np.linalg.det(self.prior_cov_theta)) + \
                        constant_terms_from_log_gaussian(self.dim_x, self.det_R_theta)
        

        init_quad_forms_in_z.append(self._get_quad_form_in_z_obs_term(observation))
        init_quad_forms_in_z.append(QuadForm(Omega=-0.5*inv(self.prior_cov_theta), b=-self.prior_mean_theta))


        return init_constants, init_quad_forms_in_z

    def compute(self, observations):

        filtering_mean, filtering_cov = self._init_filtering(observations[0])
        constants, quad_forms_in_z = self.init_V(observations[0])

        for observation in observations[1:]:

            A_backward, a_backward, backward_cov = self._update_backward(filtering_mean, filtering_cov)
            new_constants, new_quad_forms_in_z = self._expectations_under_backward(quad_forms_in_z, 
                                                                          A_backward, a_backward, backward_cov, 
                                                                          observation)

            constants += new_constants
            quad_forms_in_z = new_quad_forms_in_z

            filtering_mean, filtering_cov = self._update_filtering(observation, filtering_mean, filtering_cov)

        constants += -constant_terms_from_log_gaussian(self.dim_z, np.linalg.det(filtering_cov)) 
        constants += self._expectations_under_filtering(quad_forms_in_z, filtering_mean, filtering_cov) + 0.5*self.dim_z
        
        return constants 

                





        



        
        


