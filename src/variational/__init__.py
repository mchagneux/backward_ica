from src.stats.hmm import LinearGaussianHMM
from src.variational.sequential_models import NeuralBackwardSmoother, JohnsonBackward

def get_variational_model(args, p=None, key_for_random_params=None):

    if args.model == 'linear':
        q = LinearGaussianHMM(
                state_dim=args.state_dim, 
                obs_dim=args.obs_dim,
                transition_matrix_conditionning=args.transition_matrix_conditionning,
                range_transition_map_params=args.range_transition_map_params,
                transition_bias=args.transition_bias, 
                emission_bias=args.emission_bias)

    elif 'neural_backward' in args.model:
        if 'explicit_transition' in args.model_options:
              q = NeuralBackwardSmoother.with_linear_gaussian_transition_kernel(args)
        else:
              
              q = NeuralBackwardSmoother(
                state_dim=args.state_dim, 
                obs_dim=args.obs_dim,
                transition_kernel=None,
                backwd_layers=args.backwd_layers,
                update_layers=args.update_layers)

    elif 'johnson_backward' in args.model:
            q = JohnsonBackward.from_args(args)


    if key_for_random_params is not None:
        phi = q.get_random_params(key_for_random_params, args)
        return q, phi
    else:
        return q