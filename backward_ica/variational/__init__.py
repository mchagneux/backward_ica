from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.variational.models import NeuralLinearBackwardSmoother, JohnsonBackward, JohnsonForward

def get_variational_model(args, p=None, key_for_random_params=None):

    if args.q_version == 'linear':

        q = LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim,
                            transition_matrix_conditionning='init_sym_def_pos',
                            range_transition_map_params=(-1,1),
                            transition_bias=True, 
                            emission_bias=True)

    elif 'neural_backward_linear' in args.q_version:
        if (p is not None) and (p.transition_kernel.map_type == 'linear'):
            q = NeuralLinearBackwardSmoother.with_transition_from_p(p, args.update_layers)

        elif 'backwd_net' in args.q_version:
            q = NeuralLinearBackwardSmoother(state_dim=args.state_dim, 
                                                obs_dim=args.obs_dim,
                                                transition_kernel=None,
                                                update_layers=args.update_layers)
        else:
            q = NeuralLinearBackwardSmoother.with_linear_gaussian_transition_kernel(p, args.update_layers)
        
    # elif args.q_version == 'neural_backward':
    #     q = NeuralBackwardSmoother(state_dim=args.state_dim, 
    #                                     obs_dim=args.obs_dim, 
    #                                     update_layers=args.update_layers,
    #                                     backwd_layers=args.backwd_map_layers)

    elif 'johnson_backward' in args.q_version:
            q = JohnsonBackward(state_dim=args.state_dim, 
                                    obs_dim=args.obs_dim, 
                                    layers=args.update_layers,
                                    anisotropic=args.anisotropic)

    elif 'johnson_forward' in args.q_version:
            q = JohnsonForward(state_dim=args.state_dim, 
                                    obs_dim=args.obs_dim, 
                                    layers=args.update_layers,
                                    anisotropic=args.anisotropic)


    if key_for_random_params is not None:
        phi = q.get_random_params(key_for_random_params, args)
        return q, phi
    else:
        return q