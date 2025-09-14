# model_factory.py
import inspect
import math
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# --- Transformer를 위한 PositionalEncoding 클래스 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x의 shape은 (batch, seq_len, d_model)이므로 pe를 (seq_len, 1, d_model)에서
        # (1, seq_len, d_model)로 바꾼 후 더해줍니다.
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)

# --- Transformer 피처 추출기 ---
class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        seq_space = observation_space["sequence"]
        inst_space = observation_space["instant"]
        
        input_dim = seq_space.shape[1]
        d_model = 128  # Transformer의 내부 임베딩 차원
        nhead = 4      # Multi-head Attention의 헤드 수
        d_hid = 256    # Feedforward network의 차원
        nlayers = 2    # Transformer Encoder Layer의 수
        dropout = 0.1

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_space.shape[0])
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        combined_features_dim = d_model + inst_space.shape[0]
        self.linear = nn.Sequential(
            nn.Linear(combined_features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        sequence_data = observations["sequence"]
        instant_data = observations["instant"]
        
        # 1. 입력 프로젝션: (batch, seq, input_dim) -> (batch, seq, d_model)
        src = self.input_proj(sequence_data)
        # 2. Positional Encoding 추가
        src = self.pos_encoder(src)
        # 3. Transformer Encoder 통과
        output = self.transformer_encoder(src)
        # 4. 마지막 스텝의 출력만 사용
        last_step_output = output[:, -1, :]
        
        # 5. 인스턴스 피처와 결합
        combined_features = torch.cat([last_step_output, instant_data], dim=1)
        
        # 6. 최종 피처 벡터 반환
        return self.linear(combined_features)


class LstmFeatureExtractor(BaseFeaturesExtractor):
    """LSTM을 사용하여 시계열 데이터에서 피처를 추출하는 클래스."""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        seq_space = observation_space["sequence"]
        inst_space = observation_space["instant"]
        input_dim = seq_space.shape[1]
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=self.lstm_hidden_size,
            num_layers=2, batch_first=True, dropout=0.1
        )
        combined_features_dim = self.lstm_hidden_size + inst_space.shape[0]
        self.linear = nn.Sequential(
            nn.Linear(combined_features_dim, features_dim), nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        sequence_data = observations["sequence"]
        instant_data = observations["instant"]
        lstm_out, _ = self.lstm(sequence_data)
        last_hidden_state = lstm_out[:, -1, :]
        combined_features = torch.cat([last_hidden_state, instant_data], dim=1)
        return self.linear(combined_features)


ALGOS = {
    'ppo': PPO,
    'sac': SAC,
    'a2c': A2C,
    'ddpg': DDPG,
    'td3': TD3,
}

class ModelFactory:
    @staticmethod
    def create(args, env, tensorboard_log, logger, policy_kwargs=None):
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
            })
        elif algo == 'sac':
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
            })
        elif algo == 'a2c':
            kwargs.update({
                'gamma': args.a2c_gamma,
                'learning_rate': args.a2c_lr,
                'ent_coef': args.a2c_ent_coef,
                'vf_coef': args.a2c_vf_coef,
                'n_steps': args.a2c_n_steps,
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
            })

        # policy_kwargs 및 정책 이름 처리 로직
        final_policy_kwargs = {}
        if policy_kwargs is not None:
            # 외부에서 policy_kwargs가 주입되면 그대로 사용
            final_policy_kwargs = policy_kwargs
            policy_name = "MultiInputPolicy"
            print("INFO: Custom Feature Extractor in use. Policy set to MultiInputPolicy.")
        else:
            # 주입되지 않으면 기존 방식으로 net_arch 기반의 policy_kwargs 생성
            print(f"INFO: Using default Feature Extractor. Policy set to {args.policy}.")
            if algo == 'ppo':
                final_policy_kwargs['net_arch'] = {
                    'pi': args.ppo_net_arch_pi, 'vf': args.ppo_net_arch_vf
                }
            elif algo == 'sac':
                 final_policy_kwargs['net_arch'] = {
                    'pi': args.sac_net_arch_pi, 'qf': args.sac_net_arch_qf
                }
            elif algo == 'a2c':
                final_policy_kwargs['net_arch'] = args.a2c_net_arch
            elif algo == 'ddpg':
                final_policy_kwargs['net_arch'] = args.ddpg_net_arch
            elif algo == 'td3':
                final_policy_kwargs['net_arch'] = {
                    'pi': args.td3_net_arch_pi, 'qf': args.td3_net_arch_qf
                }
            policy_name = args.policy

        kwargs['policy_kwargs'] = final_policy_kwargs
        
        # 결정된 정책 이름과 파라미터로 모델 생성
        model = AlgoCls(policy_name, **kwargs)
        model.set_logger(logger)
        return model
