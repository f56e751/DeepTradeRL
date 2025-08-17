# model_factory.py
import inspect
from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3

ALGOS = {
    'ppo': PPO,
    'sac': SAC,
    'a2c': A2C,
    'ddpg': DDPG,
    'td3': TD3,
}

class ModelFactory:
    @staticmethod
    def create(args, env, tensorboard_log, logger):
        algo = args.algo
        AlgoCls = ALGOS[algo]
        
        # 공통 인자
        kwargs = {
            'env': env,
            'verbose': 1,
            'tensorboard_log': tensorboard_log,
            'device': args.device,
            'seed': args.seed,
        }
        
        # 알고리즘 별로 argparse에서 받은 파라미터를 꺼내서 kwargs에 추가
        if algo == 'ppo':
            kwargs.update({
                'gamma': args.ppo_gamma,
                'learning_rate': args.ppo_lr,
                'ent_coef': args.ppo_ent_coef,
                'max_grad_norm': args.ppo_max_grad_norm,
                'vf_coef': args.ppo_vf_coef,
                'policy_kwargs': {
                    'net_arch': {
                        'pi': args.ppo_net_arch_pi,
                        'vf': args.ppo_net_arch_vf
                    }
                }
            })
        elif algo == 'sac':
            # ent_coef, target_entropy 은 'auto' 처리
            ent_coef = args.sac_ent_coef if args.sac_ent_coef != 'auto' else 'auto'
            target_entropy = args.sac_target_entropy if args.sac_target_entropy != 'auto' else 'auto'
            
            kwargs.update({
                'gamma': args.sac_gamma,
                'tau': args.sac_tau,
                'ent_coef': ent_coef,
                'target_update_interval': args.sac_target_update_interval,
                'target_entropy': target_entropy,
                'use_sde': args.sac_use_sde,
                'sde_sample_freq': args.sac_sde_sample_freq,
                'use_sde_at_warmup': args.sac_use_sde_at_warmup,
                'learning_rate': args.sac_lr,
                'buffer_size': args.sac_buffer_size,
                'learning_starts': args.sac_learning_starts,
                'batch_size': args.sac_batch_size,
                'train_freq': args.sac_train_freq,
                'gradient_steps': args.sac_gradient_steps,
                'policy_kwargs': {
                    'net_arch': {
                        'pi': args.sac_net_arch_pi,   # e.g. [64,64]
                        'qf': args.sac_net_arch_qf    # e.g. [64,64]
                    }
                }
            })
            
        elif algo == 'a2c':
            kwargs.update({
                'gamma': args.a2c_gamma,
                'learning_rate': args.a2c_lr,
                'ent_coef': args.a2c_ent_coef,
                'vf_coef': args.a2c_vf_coef,
                'n_steps': args.a2c_n_steps,
                'policy_kwargs': {
                    'net_arch': args.a2c_net_arch
                }
            })

        elif algo == 'ddpg':
            kwargs.update({
                'gamma': args.ddpg_gamma,
                'learning_rate': args.ddpg_lr,
                'tau': args.ddpg_tau,
                'batch_size': args.ddpg_batch_size,
                'buffer_size': args.ddpg_buffer_size,
                'learning_starts': args.ddpg_learning_starts,
                'train_freq': args.ddpg_train_freq,
                'gradient_steps': args.ddpg_gradient_steps,
                # Action noise 같은 특별 옵션이 있다면 argparse에도 정의 후 여기에 추가
                # 'action_noise': args.ddpg_action_noise,
                'policy_kwargs': {
                    'net_arch': args.ddpg_net_arch
                }
            })

        elif algo == 'td3':
            kwargs.update({
                'gamma': args.td3_gamma,
                'learning_rate': args.td3_lr,
                'tau': args.td3_tau,
                'batch_size': args.td3_batch_size,
                'buffer_size': args.td3_buffer_size,
                'learning_starts': args.td3_learning_starts,
                'train_freq': args.td3_train_freq,
                'gradient_steps': args.td3_gradient_steps,
                'policy_kwargs': {
                    'net_arch': {
                        'pi': args.td3_net_arch_pi,
                        'qf': args.td3_net_arch_qf
                    }
                }
            })

        
        # 마지막으로 policy, logger 설정
        model = AlgoCls(args.policy, **kwargs)
        model.set_logger(logger)
        return model
