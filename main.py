# ===== IMPORTS =====

# Standard Library
import os
import gc
import copy
import random
import shutil
import time
import datetime
from pathlib import Path
import hashlib
import json

# Third-Party Libraries
import numpy as np
import pandas as pd
import psutil
import ccxt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.style as mplstyle
import talib as ta
from torch.utils.tensorboard import SummaryWriter
import cProfile
import pstats
import io

# ===== HYPERPARAMETERS =====

hyperparameters = {
    # General
    'SEED': 77,
    'SYMBOLS': ['BTC/USDT'],
    
    # Training Parameters
    'NUM_EPISODES': 100,
    'PLOT_UPDATE_INTERVAL': 128,
    'MODEL_UPDATE_INTERVAL': 1,
    
    # Agent Parameters
    'AGENT_BATCH_SIZE': 64,
    'AGENT_GAMMA': 0.99,
    'AGENT_LR': 1e-4,
    'TTT_LR': 1e-6,  # Learning rate for Test-Time Training
    'AGENT_TAU': 0.001,
    # 'LR_SCHEDULER_PATIENCE': 1000,
    # 'LR_SCHEDULER_FACTOR': 0.7,
    # 'LR_SCHEDULER_MIN_LR': 1e-7,
    # 'LR_SCHEDULER_THRESHOLD': 1e-7,
    'SIGMA_INIT': 0.618,
    'LORA_R': 8,
    'LORA_ALPHA': 1.0,
    
    # Environment Parameters
    'ENV_LOOKBACK': 32,
    'ENV_INITIAL_CAPITAL': 1000,
    'ENV_BUY_COST_PCT': 0.006, #Coinbase fees
    'ENV_SELL_COST_PCT': 0.006, #Coinbase fees
    'ENV_GAMMA': 0.99,
    'ENV_LEVERAGE': 1,
    'ENV_MAX_CAPITAL_PCT_PER_TRADE': 0.1,
    'ENV_TIMEFRAME': '1d',
    'ENV_STOP_LOSS_PCT': 0.1,

    # Dates
    'DATA_START_DATE': '2020-01-01',
    'TRAIN_START_DATE': '2021-01-01',
    'TRAIN_END_DATE': '2023-12-31',
    'TEST_BEGIN_DATE': '2024-01-02',
    'TEST_END_DATE': '2024-10-31'
}

# ===== FUNCTION DEFINITIONS =====

def generate_signature(hyperparams):
    """
    Generate a unique SHA256 hash signature from the hyperparameters.
    """
    hyperparams_json = json.dumps(hyperparams, sort_keys=True)
    signature = hashlib.sha256(hyperparams_json.encode('utf-8')).hexdigest()
    return signature

def generate_signature_from_hyperparams():
    """
    Generates a signature based on the global hyperparameters dictionary.
    """
    return generate_signature(hyperparameters)

def get_device():
    """
    Determine the available device (CUDA, MPS, or CPU) for computations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def set_seed(seed=hyperparameters['SEED']):
    """
    Set random seed for reproducibility across various libraries.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fetch_ohlcv_data(exchange, symbol, start_date, end_date, timeframe):
    """
    Fetch OHLCV data from Binance for a given symbol and date range.
    """
    since = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    ohlcv_data = []
    
    while since < end_timestamp:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not data:
                break
            ohlcv_data.extend(data)
            since = data[-1][0] + exchange.parse_timeframe(timeframe) * 1000
        except ccxt.BaseError as e:
            print(f"An error occurred: {e}")
            break
    
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%dT%H:%M')
    df['tic'] = symbol
    return df

def prepare_data(df):
    features = pd.DataFrame()
    features['timestamp'] = df['timestamp']
    features['date'] = df['date']

    # Time-based features
    dt = pd.to_datetime(df['timestamp'], unit='ms')
    features['hour'] = dt.dt.hour
    features['week_day'] = dt.dt.weekday
    features['month_day'] = dt.dt.day
    features['month'] = dt.dt.month

    # Cyclical encoding for time features
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['week_day_sin'] = np.sin(2 * np.pi * features['week_day'] / 7)
    features['week_day_cos'] = np.cos(2 * np.pi * features['week_day'] / 7)
    features['month_day_sin'] = np.sin(2 * np.pi * features['month_day'] / 31)
    features['month_day_cos'] = np.cos(2 * np.pi * features['month_day'] / 31)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

    # Drop original time features
    features = features.drop(['hour', 'week_day', 'month_day', 'month'], axis=1)

    # Price and Volume
    features[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']]

    # Technical Indicators
    # RSI
    features['rsi'] = ta.RSI(df['close']) / 100  # Normalize to [0, 1]

    # Stochastic Oscillator
    features['stoch_k'], features['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
    features['stoch_k'] /= 100  # Normalize to [0, 1]
    features['stoch_d'] /= 100

    # Stochastic RSI
    features['stochrsi_fastk'], features['stochrsi_fastd'] = ta.STOCHRSI(df['close'])
    features['stochrsi_fastk'] /= 100
    features['stochrsi_fastd'] /= 100

    # Williams %R
    features['willr'] = ta.WILLR(df['high'], df['low'], df['close'])
    features['willr'] = (features['willr'] + 100) / 100  # Normalize to [0, 1]

    # MACD (no fixed range)
    features['macd'], features['macdsignal'], features['macdhist'] = ta.MACD(df['close'])
    # We'll normalize these later

    # PPO (no fixed range)
    features['ppo'] = ta.PPO(df['close'])
    # Normalize later

    # APO (no fixed range)
    features['apo'] = ta.APO(df['close'])
    # Normalize later

    # CCI
    features['cci'] = ta.CCI(df['high'], df['low'], df['close'])
    # Normalize approximately to [-1, 1]
    features['cci'] = features['cci'] / 200

    # Momentum Indicators
    features['roc'] = ta.ROC(df['close'])  # No fixed range
    features['mom'] = ta.MOM(df['close'])  # No fixed range
    features['trix'] = ta.TRIX(df['close'])  # No fixed range

    # ADX
    features['adx'] = ta.ADX(df['high'], df['low'], df['close']) / 100
    features['adxr'] = ta.ADXR(df['high'], df['low'], df['close']) / 100

    # Aroon Oscillator
    features['aroonosc'] = ta.AROONOSC(df['high'], df['low'])
    features['aroonosc'] = (features['aroonosc'] + 100) / 200  # Normalize to [0, 1]

    # MFI
    features['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume']) / 100

    # Candlestick Patterns
    features['cdl_hammer'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close']) / 100
    features['cdl_engulfing'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) / 100

    # Clean and Prepare for Normalization
    features.dropna(inplace=True)
    features.sort_values(['timestamp'], inplace=True)
    features.reset_index(drop=True, inplace=True)

    # For indicators without fixed ranges, apply min-max normalization
    columns_to_minmax = [
        'open', 'high', 'low', 'close', 'volume',
        'macd', 'macdsignal', 'macdhist',
        'ppo', 'apo', 'roc', 'mom', 'trix',
    ]

    features[columns_to_minmax] = features[columns_to_minmax].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))

    return features

def log_memory_usage(episode):
    """
    Log the current memory usage.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB
    print(f"Episode {episode}: Memory usage: {mem:.2f} GB")

# ===== CLASS DEFINITIONS =====

class Autoencoder(nn.Module):
    """
    Simple Autoencoder for state reconstruction.
    """
    def __init__(self, input_dim, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Tanh()  # Assuming input features are normalized between -1 and 1
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class NoisyLoRALinear(nn.Module):
    """
    Linear layer with both LoRA (Low-Rank Adaptation) and Noisy Nets functionality.
    """
    def __init__(self, in_features, out_features, r=8, alpha=1.0, lora_active=False, sigma_init=0.5, bias=True):
        super(NoisyLoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.lora_active = lora_active
        self.scaling = alpha / r

        # Main weight and bias parameters (mu and sigma)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)

        # LoRA parameters
        self.A = nn.Parameter(torch.Tensor(r, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, r))

        # Noisy Nets parameters for LoRA
        self.weight_A_sigma = nn.Parameter(torch.Tensor(r, in_features))
        self.weight_B_sigma = nn.Parameter(torch.Tensor(out_features, r))

        # Noise variables (not registered as buffers)
        self.weight_epsilon = None
        self.bias_epsilon = None
        self.weight_A_epsilon = None
        self.weight_B_epsilon = None

        self.sigma_init = sigma_init
        self.reset_parameters()
        # No need to call reset_noise() here, as it will be called during the first forward pass

    def reset_parameters(self):
        # Initialize mu parameters
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)

        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init * mu_range)

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.zeros_(self.B)
        self.weight_A_sigma.data.fill_(self.sigma_init * mu_range)
        self.weight_B_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        """Reset the noise variables for Noisy Nets."""
        if self.training:
            # Generate new noise variables
            self.weight_epsilon = torch.randn_like(self.weight_mu)
            if self.bias_mu is not None:
                self.bias_epsilon = torch.randn_like(self.bias_mu)
            else:
                self.bias_epsilon = None

            if self.lora_active and self.r > 0:
                self.weight_A_epsilon = torch.randn_like(self.A)
                self.weight_B_epsilon = torch.randn_like(self.B)
            else:
                self.weight_A_epsilon = None
                self.weight_B_epsilon = None
        else:
            # Use zeros for evaluation mode
            self.weight_epsilon = torch.zeros_like(self.weight_mu)
            if self.bias_mu is not None:
                self.bias_epsilon = torch.zeros_like(self.bias_mu)
            else:
                self.bias_epsilon = None

            if self.lora_active and self.r > 0:
                self.weight_A_epsilon = torch.zeros_like(self.A)
                self.weight_B_epsilon = torch.zeros_like(self.B)
            else:
                self.weight_A_epsilon = None
                self.weight_B_epsilon = None

    def forward(self, x):
        self.reset_noise()

        # Noisy main weights
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        if self.bias_mu is not None:
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            bias = None

        # LoRA adjustment
        if self.lora_active and self.r > 0:
            weight_A = self.A + self.weight_A_sigma * self.weight_A_epsilon
            weight_B = self.B + self.weight_B_sigma * self.weight_B_epsilon
            lora_adjustment = (x @ weight_A.t()) @ weight_B.t() * self.scaling
            output = F.linear(x, weight) + lora_adjustment
        else:
            output = F.linear(x, weight)

        if bias is not None:
            output += bias

        return output

    def activate_lora(self):
        """Activate LoRA adapters."""
        self.lora_active = True

    def deactivate_lora(self):
        """Deactivate LoRA adapters."""
        self.lora_active = False


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples for replay.
    """
    def __init__(self, capacity=1024, state_shape=(hyperparameters['ENV_LOOKBACK'],), device='cpu'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.full = False
        self.state_shape = state_shape
        self.reset_buffers()

    def reset_buffers(self):
        """Initialize or reset the buffer tensors."""
        self.states = torch.zeros((self.capacity, *self.state_shape), dtype=torch.float32)
        self.actions = torch.zeros(self.capacity, dtype=torch.long)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32)
        self.next_states = torch.zeros((self.capacity, *self.state_shape), dtype=torch.float32)
        self.dones = torch.zeros(self.capacity, dtype=torch.float32)  # Keep dtype as float32

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        idx = self.position
        self.states[idx].copy_(state.detach().cpu())
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx].copy_(next_state.detach().cpu())
        self.dones[idx] = float(done)  # Convert done to float
        self.position = (self.position + 1) % self.capacity
        self.full = self.full or self.position == 0

    def sample(self, batch_size=hyperparameters['AGENT_BATCH_SIZE']):
        """Randomly sample a batch of experiences from the buffer."""
        max_mem = self.capacity if self.full else self.position
        indices = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device)
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return self.capacity if self.full else self.position

    def clear(self):
        """Clear the buffer and reset."""
        self.position = 0
        self.full = False
        del self.states, self.actions, self.rewards, self.next_states, self.dones
        self.reset_buffers()

    def __getstate__(self):
        """Serialize the replay buffer."""
        state = self.__dict__.copy()
        # Move tensors to CPU before saving
        state['states'] = self.states.cpu()
        state['actions'] = self.actions.cpu()
        state['rewards'] = self.rewards.cpu()
        state['next_states'] = self.next_states.cpu()
        state['dones'] = self.dones.cpu()
        return state

    def __setstate__(self, state):
        """Deserialize the replay buffer."""
        self.__dict__.update(state)
        # Move tensors back to the device
        self.states = self.states.to(self.device)
        self.actions = self.actions.to(self.device)
        self.rewards = self.rewards.to(self.device)
        self.next_states = self.next_states.to(self.device)
        self.dones = self.dones.to(self.device)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for Transformer models.
    """
    def __init__(self, d_model, max_len=1024):
        super(RelativePositionalEncoding, self).__init__()
        self.rel_pos_table = nn.Parameter(torch.randn(2 * max_len - 1, d_model))

    def forward(self, seq_len):
        """
        Generate relative positional embeddings.
        """
        range_vec = torch.arange(seq_len, device=self.rel_pos_table.device)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = distance_mat.clamp(-self.rel_pos_table.size(0)//2 + 1, self.rel_pos_table.size(0)//2 - 1)
        final_mat = distance_mat_clipped + self.rel_pos_table.size(0) // 2 - 1
        embeddings = self.rel_pos_table[final_mat]  # (seq_len, seq_len, d_model)
        return embeddings

class MultiheadAttentionRelative(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, lora_r=8, alpha=1.0, lora_active=False, sigma_init=hyperparameters['SIGMA_INIT']):
        super(MultiheadAttentionRelative, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Q, K, V projections with NoisyLoRALinear
        self.q_linear = NoisyLoRALinear(embed_dim, embed_dim, r=lora_r, alpha=alpha, lora_active=lora_active, sigma_init=sigma_init)
        self.k_linear = NoisyLoRALinear(embed_dim, embed_dim, r=lora_r, alpha=alpha, lora_active=lora_active, sigma_init=sigma_init)
        self.v_linear = NoisyLoRALinear(embed_dim, embed_dim, r=lora_r, alpha=alpha, lora_active=lora_active, sigma_init=sigma_init)
        self.out_proj = NoisyLoRALinear(embed_dim, embed_dim, r=lora_r, alpha=alpha, lora_active=lora_active, sigma_init=sigma_init)
        self.dropout_layer = nn.Dropout(dropout)

        self.relative_positional_encoding = RelativePositionalEncoding(self.head_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):

        seq_len, batch_size, embed_dim = query.size()
        q = self.q_linear(query).view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = self.k_linear(key).view(seq_len, batch_size, self.num_heads, self.head_dim)
        v = self.v_linear(value).view(seq_len, batch_size, self.num_heads, self.head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        q = q * self.scaling

        content_scores = torch.matmul(q, k.transpose(-2, -1))

        rel_pos_embeddings = self.relative_positional_encoding(seq_len)
        pos_scores = torch.einsum('bnsh,srh->bnsr', q, rel_pos_embeddings)

        attn_scores = content_scores + pos_scores

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        output = output.permute(1, 0, 2)

        return output


class TransformerEncoderLayerRelative(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, lora_r=8, alpha=1.0, lora_active=False, sigma_init=hyperparameters['SIGMA_INIT']):
        super(TransformerEncoderLayerRelative, self).__init__()
        self.self_attn = MultiheadAttentionRelative(d_model, nhead, dropout=dropout, lora_r=lora_r, alpha=alpha, lora_active=lora_active, sigma_init=sigma_init)

        # Feedforward network with NoisyLoRALinear
        self.linear1 = NoisyLoRALinear(d_model, dim_feedforward, r=lora_r, alpha=alpha, lora_active=lora_active, sigma_init=sigma_init)
        self.linear2 = NoisyLoRALinear(dim_feedforward, d_model, r=lora_r, alpha=alpha, lora_active=lora_active, sigma_init=sigma_init)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def reset_noise(self):
        """Reset noise in NoisyLoRALinear layers within the encoder layer."""
        self.self_attn.reset_noise()
        self.linear1.reset_noise()
        self.linear2.reset_noise()

class TransformerEncoderRelative(nn.Module):
    """
    Transformer encoder composed of multiple relative encoder layers.
    """
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoderRelative, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(encoder_layer.linear2.out_features)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Forward pass through all encoder layers.
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        output = self.norm(output)
        return output

class TransformerNetwork(nn.Module):
    def __init__(self, state_dim, output_dim, lookback, nhead=8, num_layers=4, model_dim=512, lora_r=8, lora_alpha=1.0, lora_active=False, sigma_init=hyperparameters['SIGMA_INIT']):
        super(TransformerNetwork, self).__init__()
        self.model_dim = model_dim
        self.lookback = lookback
        self.lora_active = lora_active  # Control LoRA activation

        # Input projection with NoisyLoRALinear
        self.input_fc = NoisyLoRALinear(state_dim, self.model_dim, r=lora_r, alpha=lora_alpha, lora_active=lora_active, sigma_init=sigma_init)

        # Transformer encoder layers with NoisyLoRALinear
        encoder_layer = TransformerEncoderLayerRelative(d_model=model_dim, nhead=nhead, lora_r=lora_r, alpha=lora_alpha, lora_active=lora_active, sigma_init=sigma_init)
        self.transformer = TransformerEncoderRelative(encoder_layer, num_layers=num_layers)

        # Output head with NoisyLoRALinear
        self.output_fc = NoisyLoRALinear(self.model_dim, output_dim, r=lora_r, alpha=lora_alpha, lora_active=lora_active, sigma_init=sigma_init)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        for module in self.modules():
            if isinstance(module, NoisyLoRALinear):
                module.reset_parameters()

    def forward(self, x):

        x = self.input_fc(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        output = self.output_fc(x)
        return output

    def reset_noise(self):
        """Reset noise in all NoisyLoRALinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLoRALinear):
                module.reset_noise()


class DiscreteDDQNAgent:
    """
    Discrete Double DQN Agent with NoisyNet and LoRA for TTT.
    """
    ACTION_OPEN_LONG = 0
    ACTION_OPEN_SHORT = 1
    ACTION_CLOSE_LONG = 2
    ACTION_CLOSE_SHORT = 3

    def __init__(
        self,
        state_dim,
        action_dim,
        lookback,
        batch_size=hyperparameters['AGENT_BATCH_SIZE'],
        gamma=hyperparameters['AGENT_GAMMA'],
        lr=hyperparameters['AGENT_LR'],
        tau=hyperparameters['AGENT_TAU'],
        device='cpu',
        lora_r=8,
        lora_alpha=1.0,
        sigma_init=hyperparameters['SIGMA_INIT']
    ):
        self.sig = generate_signature_from_hyperparams()
        self.writer  = SummaryWriter(log_dir=f'runs/iansan_{self.sig}')
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.lora_active = False  # LoRA is inactive during regular training
        self.in_ttt = False

        self.online_net = TransformerNetwork(state_dim, action_dim, lookback, lora_r=lora_r, lora_alpha=lora_alpha, lora_active=self.lora_active, sigma_init=sigma_init).to(self.device)
        self.target_net = TransformerNetwork(state_dim, action_dim, lookback, lora_r=lora_r, lora_alpha=lora_alpha, lora_active=self.lora_active, sigma_init=sigma_init).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, weight_decay=1e-5)

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=hyperparameters['LR_SCHEDULER_FACTOR'], patience=hyperparameters['LR_SCHEDULER_PATIENCE'], min_lr=hyperparameters['LR_SCHEDULER_MIN_LR'], threshold=hyperparameters['LR_SCHEDULER_THRESHOLD'])

        self.loss_fn = nn.HuberLoss()

        self.memory = ReplayBuffer(capacity=1024, state_shape=(lookback, state_dim), device=self.device)

        self.iteration = 0

        # Initialize the Autoencoder
        input_dim = lookback * state_dim
        self.autoencoder = Autoencoder(input_dim=input_dim, latent_dim=128).to(self.device)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        state_detached = state.squeeze(0).detach().cpu()
        next_state_detached = next_state.squeeze(0).detach().cpu()
        self.memory.push(state_detached, action, reward, next_state_detached, done)

    def choose_action(self, state):
        """
        Choose an action based on the maximum predicted Q-value.
        """
        #make sure noise is activated
        self.online_net.train()
        with torch.no_grad():
            q_values = self.online_net(state)
            # Log Q-values for analysis
            self.writer.add_scalar('Q-Values/Open Long', q_values[0][0].item(), self.iteration)
            self.writer.add_scalar('Q-Values/Open Short', q_values[0][1].item(), self.iteration)
            self.writer.add_scalar('Q-Values/Close Long', q_values[0][2].item(), self.iteration)
            self.writer.add_scalar('Q-Values/Close Short', q_values[0][3].item(), self.iteration)
            action = q_values.argmax(dim=1).item()
        return action

    def update(self):
        self.online_net.train()
        """Update the agent's networks using a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Compute Q(s, a)
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN target computation
            next_q_values = self.target_net(next_states).gather(1, self.online_net(next_states).argmax(dim=1).unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute losses
        ddqn_loss = self.loss_fn(q_values, target_q_values.detach())

        # Backpropagation for DDQN
        self.optimizer.zero_grad()
        ddqn_loss.backward()

        # Compute gradient norm for logging
        total_norm = 0
        for p in self.online_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # self.scheduler.step(ddqn_loss)
 
        # Log the learning rate
        # current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning Rate', current_lr, self.iteration)

        # Soft update the target network
        self.soft_update()

        self.writer.add_scalar('Loss/DDQN', ddqn_loss.item(), self.iteration)
        self.writer.add_scalar('Gradient Norm', total_norm, self.iteration)
    
        self.iteration += 1
        return ddqn_loss.item()

    def soft_update(self):
        """Soft update the target network parameters."""
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def activate_lora(self):
        """
        Activate LoRA adapters and freeze base model parameters.
        Only LoRA parameters will be trained during TTT.
        """
        if not self.in_ttt:
            print("Activating LoRA for Test-Time Training (TTT).")
            self.lora_active = True
            self.online_net.lora_active = True
            self.target_net.lora_active = True

            # Freeze base model parameters
            for name, param in self.online_net.named_parameters():
                if 'A' not in name and 'B' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # Re-initialize optimizer to train only LoRA parameters
            self.ttt_optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.online_net.parameters()),
                lr=hyperparameters['TTT_LR']
            )
            self.in_ttt = True

    def deactivate_lora(self):
        """
        Deactivate LoRA adapters and unfreeze all model parameters.
        """
        if self.in_ttt:
            print("Deactivating LoRA. All model parameters are now trainable.")
            self.lora_active = False
            self.online_net.lora_active = False
            self.target_net.lora_active = False

            # Unfreeze all model parameters
            for param in self.online_net.parameters():
                param.requires_grad = True

            # Re-initialize optimizer to train all parameters
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=hyperparameters['AGENT_LR'])
            self.in_ttt = False

    def update_auxiliary(self, state):
        """
        Update the Autoencoder during TTT using auxiliary loss.
        """
        self.autoencoder.train()
        self.online_net.train()  # Ensure noise and LoRA are active during TTT

        # Create masked inputs
        masked_state, target_state = self.mask_input(state)  # Both have shape (1, lookback, state_dim)

        # Flatten the states for the Autoencoder
        masked_state_flat = masked_state.view(state.size(0), -1)  # Shape: (1, lookback * state_dim)
        target_state_flat = target_state.view(state.size(0), -1)  # Shape: (1, lookback * state_dim)

        # Forward pass through Autoencoder
        reconstructed = self.autoencoder(masked_state_flat)

        # Compute auxiliary loss (MSE)
        aux_loss = F.mse_loss(reconstructed, target_state_flat)

        # Zero gradients for both optimizers
        self.autoencoder_optimizer.zero_grad()
        self.ttt_optimizer.zero_grad()

        # Backpropagation for both Autoencoder and LoRA parameters
        aux_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.online_net.parameters()), max_norm=1.0)

        # Update both optimizers
        self.autoencoder_optimizer.step()
        self.ttt_optimizer.step()

        self.autoencoder.eval()
        return aux_loss.item()

    def mask_input(self, state, mask_ratio=0.15):
        """
        Randomly mask a fraction of the input features.
        """
        batch_size, seq_len, feature_dim = state.size()
        mask = torch.rand(batch_size, seq_len, feature_dim, device=state.device) < mask_ratio
        masked_state = state.clone()
        masked_state[mask] = 0  # Replace masked positions with zeros
        target_state = state.clone()
        return masked_state, target_state

class TradingEnv:
    """
    Trading Environment for the agent to interact with.
    """
    def __init__(
        self,
        config,
        lookback=hyperparameters['ENV_LOOKBACK'],
        initial_capital=hyperparameters['ENV_INITIAL_CAPITAL'],
        buy_cost_pct=hyperparameters['ENV_BUY_COST_PCT'],
        sell_cost_pct=hyperparameters['ENV_SELL_COST_PCT'],
        gamma=hyperparameters['ENV_GAMMA'],
        leverage=hyperparameters['ENV_LEVERAGE'],
        max_capital_pct_per_trade=hyperparameters['ENV_MAX_CAPITAL_PCT_PER_TRADE'],
        device='cpu',
        timeframe=hyperparameters['ENV_TIMEFRAME'],
        stop_loss_pct=hyperparameters['ENV_STOP_LOSS_PCT'],
        start_date=None  # Accept start_date parameter
    ):
        # Environment parameters
        self.timeframe = timeframe
        self.device = device
        self.initial_capital = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.leverage = leverage
        self.max_capital_pct_per_trade = max_capital_pct_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.price_array = config["data"]
        self.tech_array = config["features"]
        self.lookback = lookback
        self.start_date = start_date

        # State variables
        self.time = 0
        self.total_steps = len(self.tech_array)
        self.cash = initial_capital
        self.free_cash = self.cash
        self.total_asset = initial_capital
        self.stocks_long = 0
        self.stocks_short = 0
        self.long_entry_price = 0
        self.short_entry_price = 0
        self.long_position_duration = 0
        self.short_position_duration = 0
        self.account_values = [self.total_asset]
        self.benchmark_values = [self.initial_capital]
        
        # Precompute data for faster access
        self.precomputed_price, self.precomputed_tech = self.precompute_data()

        # Environment configuration
        feature_size = len(self.tech_array.columns) - 2  # Exclude 'timestamp' and 'date'
        self.state_dim = 2 + feature_size  # Include weight_long and weight_short
        self.action_dim = 4  # Actions: 0-Open Long, 1-Open Short, 2-Close Long, 3-Close Short
        self.lookback = lookback

        # Determine starting index based on start_date
        if self.start_date is not None:
            start_indices = self.tech_array.index[self.tech_array['date'] >= self.start_date]
            if len(start_indices) > 0:
                self.start_index = start_indices[0]
            else:
                self.start_index = 0
        else:
            self.start_index = 0

        # Adjust max_step accordingly
        self.max_step = len(self.tech_array) - self.start_index - 1

        # Metrics for tracking performance
        self.max_drawdown = 1
        self.max_total_asset = self.total_asset
        self.num_winning_trades = 0
        self.num_losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.gains = []
        self.losses = []
        self.rewards = []
        self.episode_data = []

        # Initialize plotting
        self.init_plot()

    def precompute_data(self):
        """Precompute data for faster access during training."""
        precomputed_price = {row['timestamp']: row for _, row in self.price_array.iterrows()}
        precomputed_tech = {row['timestamp']: row for _, row in self.tech_array.iterrows()}
        return precomputed_price, precomputed_tech

    def reset(self):
        """Reset the environment state for a new episode."""
        self.time = self.start_index  # Start from the computed start index
        self.cash = self.initial_capital
        self.free_cash = self.cash
        self.total_asset = self.initial_capital
        self.stocks_long = 0
        self.stocks_short = 0
        self.total_profit = 0
        self.total_loss = 0
        self.long_entry_price = 0
        self.short_entry_price = 0
        self.long_position_duration = 0
        self.short_position_duration = 0
        self.episode_data = []
        self.account_values = [self.total_asset]
        self.benchmark_values = [self.initial_capital]

        # Reset metrics
        self.max_drawdown = 1
        self.max_total_asset = self.total_asset
        self.num_winning_trades = 0
        self.num_losing_trades = 0
        self.gains = []
        self.losses = []
        self.rewards = []

        # Initialize current price
        current_timestamp = self.tech_array.iloc[self.time]['timestamp']
        price_row = self.precomputed_price.get(current_timestamp)
        if price_row is not None:
            self.current_price = price_row['close']
            self.current_low = price_row['low']
            self.current_high = price_row['high']
        else:
            self.current_price = 0.0
            self.current_low = 0.0
            self.current_high = 0.0

        return self.get_state()

    def get_timeframe_in_seconds(self):
        """Convert timeframe string to seconds."""
        timeframe_map = {
            '1m': 60,
            '5m': 60 * 5,
            '15m': 60 * 15,
            '30m': 60 * 30,
            '1h': 60 * 60,
            '4h': 60 * 60 * 4,
            '1d': 60 * 60 * 24
        }
        return timeframe_map.get(self.timeframe, None)

    def get_periods_per_year(self):
        """Calculate the number of periods per year based on the timeframe."""
        timeframe_to_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }

        minutes_per_period = timeframe_to_minutes.get(self.timeframe)
        if minutes_per_period is None:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")

        periods_per_day = (24 * 60) / minutes_per_period
        periods_per_year = periods_per_day * 365  # Assuming 365 days for crypto
        return periods_per_year

    def get_state(self):
        """Construct the state representation for the current time step."""
        states = []
        current_timestamp = self.tech_array.iloc[self.time]['timestamp']
        weight_long = float(self.stocks_long * self.current_price / self.total_asset) if self.total_asset > 0 else 0.0
        weight_short = float(self.stocks_short * self.current_price / self.total_asset) if self.total_asset > 0 else 0.0

        for _ in range(self.lookback):
            row = self.precomputed_tech.get(current_timestamp)
            if row is not None:
                features = row.drop(['timestamp', 'date']).astype(np.float32).values
                state_row = np.concatenate(([weight_long, weight_short], features)).astype(np.float32)
            else:
                state_row = np.zeros(self.state_dim, dtype=np.float32)
            states.append(state_row)
            current_timestamp -= self.get_timeframe_in_seconds()

        states = states[::-1]  # Chronological order
        state = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(self.device)
        return state

    def step(self, discrete_action):
        """
        Execute one time step within the environment.
        Returns:
            next_state, reward, done, info
        """
        prev_total_asset = self.total_asset
        stopped = False
        done = self.time >= self.max_step
        current_timestamp = self.tech_array.iloc[self.time]['timestamp']
        price_row = self.precomputed_price.get(current_timestamp, {})
        self.current_price = price_row.get('close', self.current_price)
        self.current_low = price_row.get('low', self.current_low)
        self.current_high = price_row.get('high', self.current_price)

        # Record info for plotting and metrics
        info = {
            "date": price_row.get('date'),
            "open": price_row.get('open'),
            "high": price_row.get('high'),
            "low": price_row.get('low'),
            "close": price_row.get('close'),
            "volume": price_row.get('volume'),
            "actions": [0, 0, 0, 0],  # To record actions taken
        }

        # Check for stops
        if self.stocks_long > 0:
            stop_loss_price = self.long_entry_price * (1 - self.stop_loss_pct)
            if self.current_low <= stop_loss_price:
                stopped = True
                # Close long position at stop loss price
                exit_price = stop_loss_price
                profit = self.stocks_long * (exit_price - self.long_entry_price)
                fee = self.stocks_long * exit_price * self.sell_cost_pct
                proceeds = profit - fee
                self.cash += proceeds
                self.free_cash += self.stocks_long * exit_price
                self.stocks_long = 0
                self.long_position_duration = 0
                self.long_entry_price = 0
                info['actions'][2] = 1
                self._update_trade_metrics(profit)

        if self.stocks_short > 0:
            stop_loss_price = self.short_entry_price * (1 + self.stop_loss_pct)
            if self.current_high >= stop_loss_price:
                stopped = True
                # Close short position at stop loss price
                exit_price = stop_loss_price
                profit = self.stocks_short * (self.short_entry_price - exit_price)
                fee = self.stocks_short * exit_price * self.buy_cost_pct
                proceeds = profit - fee
                self.cash += proceeds
                self.free_cash += self.stocks_short * exit_price
                self.stocks_short = 0
                self.short_position_duration = 0
                self.short_entry_price = 0
                info['actions'][3] = 1
                self._update_trade_metrics(profit)

        # Execute actions
        if discrete_action == DiscreteDDQNAgent.ACTION_CLOSE_LONG and self.stocks_long > 0:
            profit = self.stocks_long * (self.current_price - self.long_entry_price)
            fee = self.stocks_long * self.current_price * self.sell_cost_pct
            proceeds = profit - fee
            self.cash += proceeds
            self.free_cash += self.stocks_long * self.current_price
            self.stocks_long = 0
            self.long_position_duration = 0
            self.long_entry_price = 0
            info['actions'][2] = 1
            self._update_trade_metrics(profit)

        elif discrete_action == DiscreteDDQNAgent.ACTION_CLOSE_SHORT and self.stocks_short > 0:
            profit = self.stocks_short * (self.short_entry_price - self.current_price)
            fee = self.stocks_short * self.current_price * self.buy_cost_pct
            proceeds = profit - fee
            self.cash += proceeds
            self.free_cash += self.stocks_short * self.current_price
            self.stocks_short = 0
            self.short_position_duration = 0
            self.short_entry_price = 0
            info['actions'][3] = 1
            self._update_trade_metrics(profit)

        if discrete_action == DiscreteDDQNAgent.ACTION_OPEN_LONG:
            opened = self._open_long_position()
            if opened:
                info['actions'][0] = 1

        elif discrete_action == DiscreteDDQNAgent.ACTION_OPEN_SHORT:
            opened = self._open_short_position()
            if opened:
                info['actions'][1] = 1

        # Increment position durations
        if self.stocks_long > 0:
            self.long_position_duration += 1
        if self.stocks_short > 0:
            self.short_position_duration += 1

        # Calculate PnL
        long_pnl = self.stocks_long * (self.current_price - self.long_entry_price) if self.stocks_long > 0 else 0
        short_pnl = self.stocks_short * (self.short_entry_price - self.current_price) if self.stocks_short > 0 else 0

        # Update total asset value
        self.total_asset = self.cash + long_pnl + short_pnl

        # Check for significant drawdown
        if self.total_asset <= self.initial_capital * 0.2:
            reward = -1
            done = True
            self.total_asset = 0
        else:
            if self.total_asset > self.max_total_asset:
                self.max_total_asset = self.total_asset
            drawdown = self.total_asset / self.max_total_asset
            if drawdown < self.max_drawdown:
                self.max_drawdown = drawdown

            # Calculate profit change
            profit_change = self.total_asset - prev_total_asset
            self.account_values.append(self.total_asset)
            self.rewards.append(profit_change)

            # Base reward: normalized profit change
            reward = profit_change / (prev_total_asset + 1e-8)

            # Inactivity penalty
            if self.stocks_long == 0 and self.stocks_short == 0:
                reward -= 0.001

            # Close incentive
            if discrete_action in [DiscreteDDQNAgent.ACTION_CLOSE_LONG, DiscreteDDQNAgent.ACTION_CLOSE_SHORT]:
                reward += 0.002  # Small bonus for closing

            if stopped:
                reward -= 0.1

        # Record episode data for plotting
        self.episode_data.append({
            'date': info['date'],
            'open': info['open'],
            'high': info['high'],
            'low': info['low'],
            'close': info['close'],
            'volume': info['volume'],
            'actions': info['actions'],
            'total_asset': self.total_asset,
            'stocks_long': self.stocks_long,
            'stocks_short': self.stocks_short,
            'long_entry_price': self.long_entry_price,
            'short_entry_price': self.short_entry_price,
        })

        # Increment time step if episode is not done
        if not done:
            self.time += 1

        # Get next state
        next_state = self.get_state()

        reward = np.clip(reward, -1, 1)

        return next_state, reward, done, info

    def _open_long_position(self):
        """Open a long position if possible."""
        if self.free_cash >= self.total_asset * self.max_capital_pct_per_trade:
            amount_to_invest = self.total_asset * self.max_capital_pct_per_trade
            num_shares = amount_to_invest / (self.current_price * (1 + self.buy_cost_pct))
            self.free_cash -= amount_to_invest / self.leverage
            self.stocks_long += num_shares
            if self.long_entry_price == 0:
                self.long_entry_price = self.current_price
            return True
        else:
            return False

    def _open_short_position(self):
        """Open a short position if possible."""
        if self.free_cash >= self.total_asset * self.max_capital_pct_per_trade:
            amount_to_invest = self.total_asset * self.max_capital_pct_per_trade
            num_shares = amount_to_invest / (self.current_price * (1 + self.sell_cost_pct))
            self.free_cash -= amount_to_invest / self.leverage
            self.stocks_short += num_shares
            if self.short_entry_price == 0:
                self.short_entry_price = self.current_price
            return True
        else:
            return False

    def _update_trade_metrics(self, profit):
        """Update trade metrics after closing a position."""
        if profit > 0:
            self.num_winning_trades += 1
            self.gains.append(profit)
            self.total_profit += profit
        else:
            self.num_losing_trades += 1
            self.losses.append(profit)
            self.total_loss += profit

    def init_plot(self):
        """Initialize the plot for live updating during training or testing."""
        plt.ion()
        self.fig, self.ax = plt.subplots(
            4, 1,
            figsize=(12, 8),
            num='Trading Environment',
            gridspec_kw={'height_ratios': [8, 4, 4, 1]},
            constrained_layout=True,
            clear=True
        )
        
        # Initial plots with empty data
        self.line, = self.ax[0].plot([], [], label='Close Price', color='b')
        self.asset_line, = self.ax[1].plot([], [], label='Total Asset', color='black')
        self.long_line, = self.ax[2].plot([], [], label='Stocks Long', linestyle='--', color='g')
        self.short_line, = self.ax[2].plot([], [], label='Stocks Short', linestyle='--', color='r')

        # Initialize scatter plots for annotations
        self.long_markers = self.ax[0].scatter([], [], color='green', marker='^', label='Open Long')
        self.short_markers = self.ax[0].scatter([], [], color='red', marker='v', label='Open Short')
        self.close_long_markers = self.ax[0].scatter([], [], color='green', marker='x', label='Close Long')
        self.close_short_markers = self.ax[0].scatter([], [], color='red', marker='x', label='Close Short')

        # Configure axes
        self.ax[0].set_title('Close Prices and Actions')
        self.ax[0].set_ylabel('Price')
        self.ax[0].tick_params(axis='x', rotation=10)
        self.ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%dT%H:%M'))
        self.ax[0].legend(loc='upper left')

        for ax in self.ax[1:3]:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        self.ax[1].set_title('Total Account Value')
        self.ax[1].set_ylabel('Value')
        self.ax[1].legend(loc='upper left')

        self.ax[2].set_title('Long and Short Positions')
        self.ax[2].set_ylabel('Size')
        self.ax[2].legend(loc='upper left')

        self.ax[3].axis('off')  # Hide axis for metrics

    def update_plot(self):
        """Update the plot with new data."""
        df = pd.DataFrame(self.episode_data)
        if df.empty:
            return

        # Convert date column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        df['date_num'] = mdates.date2num(df['date'])

        # Update line data
        self.line.set_data(df['date_num'], df['close'])
        self.asset_line.set_data(df['date_num'], df['total_asset'])
        self.long_line.set_data(df['date_num'], df['stocks_long'])
        self.short_line.set_data(df['date_num'], df['stocks_short'])

        # Update axis limits
        x_limits = (df['date_num'].min(), df['date_num'].max())
        for ax in self.ax[:3]:
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(x_limits)

        # Update markers for actions
        markers = [
            self.long_markers,
            self.short_markers,
            self.close_long_markers,
            self.close_short_markers
        ]

        for i, marker in enumerate(markers):
            action_dates = df['date_num'][df['actions'].apply(lambda x: x[i] > 0)].to_numpy()
            action_prices = df['close'][df['actions'].apply(lambda x: x[i] > 0)].to_numpy()
            if action_dates.size > 0:
                marker.set_offsets(np.column_stack((action_dates, action_prices)))
            else:
                marker.set_offsets(np.empty((0, 2)))

        # Update metrics in the fourth subplot
        mean_gain = float(np.mean(self.gains)) if self.gains else 0.0
        mean_loss = float(np.mean(self.losses)) if self.losses else 0.0
        drawdown_percent = float((1 - self.max_drawdown) * 100)
        total_reward = float(sum(self.rewards))
        max_total_asset = float(self.max_total_asset)

        metrics_text = (
            f"Total Reward: {total_reward:.2f}\n"
            f"Max Total Asset: {max_total_asset:.2f} | "
            f"Winning Trades: {self.num_winning_trades} | "
            f"Mean Gain: {mean_gain:.2f}\n"
            f"Max Drawdown: {drawdown_percent:.2f}% | "
            f"Losing Trades: {self.num_losing_trades} | "
            f"Mean Loss: {mean_loss:.2f}"
        )
        self.ax[3].clear()
        self.ax[3].text(
            0.05, 0.5, metrics_text, fontsize=12, va='center', ha='left', transform=self.ax[3].transAxes
        )
        self.ax[3].axis('off')

        # Refresh canvas
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        del df
        gc.collect()

    def clear_plot(self):
        """Clear the current plot."""
        self.line.set_data([], [])
        self.asset_line.set_data([], [])
        self.long_line.set_data([], [])
        self.short_line.set_data([], [])
        self.long_markers.set_offsets(np.empty((0, 2)))
        self.short_markers.set_offsets(np.empty((0, 2)))
        self.close_long_markers.set_offsets(np.empty((0, 2)))
        self.close_short_markers.set_offsets(np.empty((0, 2)))
        self.ax[3].clear()  # Clear metrics text
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        gc.collect()

    def close_plot(self):
        """Close the plot."""
        self.fig.clf()
        plt.close("all")
        gc.collect()

# ===== TRAINING AND TESTING FUNCTIONS =====

def train_agents(env, discrete_agent, symbol, episodes=hyperparameters['NUM_EPISODES'], timeframe=hyperparameters['ENV_TIMEFRAME'], device='cpu', hyperparams=None):
    """
    Train the agent over a specified number of episodes for a given symbol.
    Supports resuming training based on hyperparameter signature.
    """
    symbol_clean = ''.join(e for e in symbol if e.isalnum())
   
   # Generate signature
    signature = generate_signature(hyperparams)
    
    
    # Create directories for weights and reports specific to the symbol and timeframe
    weights_base_dir = Path('weights') / f"{symbol_clean}_{signature}"
    weights_base_dir.mkdir(parents=True, exist_ok=True)

    
    reports_dir = Path('reports') / f"{symbol_clean}_{signature}" / 'train'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training Signature: {signature}")
    
    # Directory for this specific training run
    weights_dir = weights_base_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if a training run with this signature already exists
    checkpoint_file = weights_dir / 'checkpoint.pth'
    start_episode = 1
    
    if checkpoint_file.exists():
        print(f"Resuming training from checkpoint: {checkpoint_file}")
        # Load the state_dict
        torch.serialization.add_safe_globals([ReplayBuffer])
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        # Load online_net state
        discrete_agent.online_net.load_state_dict(checkpoint['online_net_state_dict'], strict=False)
        # Load target_net state
        discrete_agent.target_net.load_state_dict(checkpoint['target_net_state_dict'], strict=False)
        # Load optimizers
        discrete_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       
        # Load replay buffer
        discrete_agent.memory = checkpoint['replay_buffer']
        start_episode = checkpoint['episode'] + 1
        print(f"Resuming from episode {start_episode}")
    else:
        print("Starting new training run.")
    
    env.init_plot()
    
    for episode in range(start_episode, episodes + 1):
        state = env.reset()
        done = False
        start_time = time.time()
        env.fig.canvas.manager.set_window_title(f"Episode: {episode} for {symbol}")
        ddqn_losses = []

        while not done:
            discrete_action = discrete_agent.choose_action(state)
            next_state, reward, done, _ = env.step(discrete_action)
            discrete_agent.store_transition(state, discrete_action, reward, next_state, done)

            if env.time % hyperparameters['MODEL_UPDATE_INTERVAL'] == 0:
                ddqn_loss = discrete_agent.update()
                if ddqn_loss is not None:
                    ddqn_losses.append(ddqn_loss)

            state = next_state

            if env.time % hyperparameters['PLOT_UPDATE_INTERVAL'] == 0:
                env.update_plot()

        # End of episode
        end_time = time.time()
        episode_duration = end_time - start_time
        avg_ddqn_loss = np.mean(ddqn_losses) if ddqn_losses else 0
        total_reward = float(sum(env.rewards))

        discrete_agent.writer.add_scalar("Reward", total_reward, episode)

        print(f"Symbol: {symbol} | Episode {episode}/{episodes} completed. Total Reward: {total_reward:.2f}, "
              f"Mean DDQN Loss: {avg_ddqn_loss:.4f} | "
              f"Duration: {episode_duration:.2f} seconds")

        # Save plot
        plot_filename = f'episode_{episode}.png'
        plot_path = reports_dir / plot_filename
        env.update_plot()
        env.fig.canvas.print_figure(str(plot_path))
        env.clear_plot()

        # Save checkpoint after each episode
        checkpoint = {
            'episode': episode,
            'online_net_state_dict': discrete_agent.online_net.state_dict(),
            'target_net_state_dict': discrete_agent.target_net.state_dict(),
            'optimizer_state_dict': discrete_agent.optimizer.state_dict(),
            'replay_buffer': discrete_agent.memory
        }
        torch.save(checkpoint, checkpoint_file)

        # Optionally, save weights if total assets doubled
        if env.total_asset >= 2 * env.initial_capital:
            episode_dir = weights_dir / f'episode_{episode}'
            episode_dir.mkdir(parents=True, exist_ok=True)
            torch.save(discrete_agent.online_net.state_dict(), episode_dir / 'discrete_online_net.pth')
            print(f"Symbol: {symbol} | Agents' weights saved at episode {episode} as total assets doubled the initial capital.")

        log_memory_usage(episode)

        # Garbage collection and CUDA cache clearance
        gc.collect()
        torch.cuda.empty_cache()

    env.close_plot()

def test_and_select_rc(test_env, discrete_agent, symbol, timeframe=hyperparameters['ENV_TIMEFRAME']):
    """
    Test the agent and select episodes based on performance for a given symbol.
    """
    symbol_clean = ''.join(e for e in symbol if e.isalnum())

    weights_dir = Path('weights')
    

    # List all signature directories
    signature_dirs = [d for d in weights_dir.iterdir() if d.is_dir()]

    for signature_dir in signature_dirs:
        symbol_from_dir = signature_dir.name.split('_')[0]
        if symbol_from_dir != symbol_clean:
            continue
        signature = signature_dir.name.split('_')[-1]
        rc_dir = Path('rc') / f"{symbol_clean}_{signature}"
        rc_dir.mkdir(parents=True, exist_ok=True)

        reports_dir = Path('reports') / f"{symbol_clean}_{signature}" / 'test'
        reports_dir.mkdir(parents=True, exist_ok=True)

        episodes = sorted([d for d in signature_dir.iterdir() if d.is_dir()], key=lambda x: int(''.join(filter(str.isdigit, x.name))))

        for episode_folder in episodes:
            episode_number = ''.join(filter(str.isdigit, episode_folder.name))

            test_env.init_plot()
            test_env.fig.canvas.manager.set_window_title(f"Episode: {episode_number} for {symbol}")

            # Load agent's weights
            online_net_path = episode_folder / 'discrete_online_net.pth'
            if online_net_path.exists():
                # Load the state_dict
                state_dict = torch.load(online_net_path, map_location=test_env.device, weights_only=True)
                # Load the state_dict into the model
                try:
                    discrete_agent.online_net.load_state_dict(state_dict, strict=False)
                except RuntimeError as e:
                    print(f"Error loading state_dict for {episode_folder.name}: {e}")
                    print("Attempting to load with strict=False.")
                    discrete_agent.online_net.load_state_dict(state_dict, strict=False)
            else:
                print(f"Weight file not found for {episode_folder.name}, skipping.")
                continue

            # Activate LoRA for TTT
            discrete_agent.activate_lora()

            # Reset test environment
            state = test_env.reset()
            done = False
            total_reward = 0

            while not done:
                # Test-Time Training: Update LoRA parameters
                aux_loss = discrete_agent.update_auxiliary(state)

                # Predict and execute action with updated online_net
                discrete_agent.online_net.eval()  # Deactivate noise during testings
                with torch.no_grad():
                    q_values = discrete_agent.online_net(state)
                    action = q_values.argmax(dim=1).item()

                # Step in the environment
                next_state, reward, done, _ = test_env.step(action)
                total_reward += reward
                state = next_state

                if test_env.time % 24 == 0:
                    test_env.update_plot()

            total_reward = float(sum(test_env.rewards))
            print(f"Symbol: {symbol} | Tested {episode_folder.name}: Total Reward: {total_reward:.2f}")
            print(f"Auxiliary Loss during TTT: {aux_loss:.4f}")

            # Save plot
            plot_filename = f"{episode_folder.name}.png"
            plot_path = reports_dir / plot_filename
            test_env.update_plot()
            test_env.fig.canvas.print_figure(str(plot_path))
            test_env.clear_plot()

            # Move weights to RC if reward threshold met
            if total_reward >= test_env.initial_capital * 0.2:
                rc_episode_dir = rc_dir / episode_folder.name
                rc_episode_dir.mkdir(parents=True, exist_ok=True)
                for file in episode_folder.iterdir():
                    shutil.copy2(file, rc_episode_dir / file.name)
                print(f"Symbol: {symbol} | Episode {episode_folder.name} moved to RC.")
            else:
                print(f"Symbol: {symbol} | Episode {episode_folder.name} not selected for RC.")

            # Deactivate LoRA after testing
            discrete_agent.deactivate_lora()

def save_experiment(symbol, timeframe, signature):
    """
    Save the experiment results by moving reports and rc folders,
    deleting weights folder, and saving hyperparameters.
    """
    symbol_clean = ''.join(e for e in symbol if e.isalnum())

    # Create a unique experiment directory
    experiment_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = Path('experiments') / f"{symbol_clean}_{timeframe}_{signature}_{experiment_time}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Move reports and rc directories into the experiment directory
    reports_src = Path('reports') / f"{symbol_clean}_{signature}"
    rc_src = Path('rc') / f"{symbol_clean}_{signature}"

    if reports_src.exists():
        shutil.move(str(reports_src), str(experiment_dir / 'reports'))
    if rc_src.exists():
        shutil.move(str(rc_src), str(experiment_dir / 'rc'))

    # Delete weights directory for this signature
    weights_dir = Path('weights') /f"{symbol_clean}_{signature}"
    if weights_dir.exists():
        shutil.rmtree(weights_dir)

    # Save hyperparameters to a txt file
    hyperparameters_file = experiment_dir / 'hyperparameters.txt'
    with open(hyperparameters_file, 'w') as f:
        f.write("Hyperparameters used in this experiment:\n")
        for key, value in hyperparameters.items():
            f.write(f"{key} = {value}\n")
        # Include any other hyperparameters or settings as needed

    print(f"Experiment saved in {experiment_dir}")

# ===== MAIN FUNCTION =====

def main():
    """
    Main function to execute training and testing of the trading agent for multiple symbols.
    """
    mplstyle.use(['ggplot', 'fast'])

    device = get_device()
    set_seed(hyperparameters['SEED'])

    # Iterate over each symbol in the SYMBOLS list
    for symbol in hyperparameters['SYMBOLS']:
        print(f"\n=== Processing Symbol: {symbol} ===\n")
        timeframe = hyperparameters['ENV_TIMEFRAME']

        # Convert date strings to datetime objects
        data_start_date_dt = datetime.datetime.strptime(hyperparameters['DATA_START_DATE'], '%Y-%m-%d')
        train_start_date_dt = datetime.datetime.strptime(hyperparameters['TRAIN_START_DATE'], '%Y-%m-%d')
        train_end_date_dt = datetime.datetime.strptime(hyperparameters['TRAIN_END_DATE'], '%Y-%m-%d')
        test_begin_date_dt = datetime.datetime.strptime(hyperparameters['TEST_BEGIN_DATE'], '%Y-%m-%d')
        test_end_date_dt = datetime.datetime.strptime(hyperparameters['TEST_END_DATE'], '%Y-%m-%d')

        # Compute lookback days based on timeframe
        timeframe_days_map = {
            '1m': 1 / (24 * 60),
            '5m': 5 / (24 * 60),
            '15m': 15 / (24 * 60),
            '30m': 30 / (24 * 60),
            '1h': 1 / 24,
            '4h': 4 / 24,
            '1d': 1
        }
        timeframe_days = timeframe_days_map.get(timeframe, 1)  # Default to 1 day if not found
        lookback_days = int(hyperparameters['ENV_LOOKBACK'] * timeframe_days)
        if lookback_days < 1:
            lookback_days = 1  # Ensure at least one day

        # Adjust data start dates to include lookback data
        train_start_date_for_data = train_start_date_dt - datetime.timedelta(days=lookback_days)
        test_start_date_for_data = test_begin_date_dt - datetime.timedelta(days=lookback_days)

        # Ensure that data_start_date_dt is early enough
        if data_start_date_dt > train_start_date_for_data:
            data_start_date_dt = train_start_date_for_data

        # Update data fetching start date
        data_start_date_str = data_start_date_dt.strftime('%Y-%m-%d')

        # Fetch data from updated start date
        exchange = ccxt.binance()
        data_path = Path.cwd() / "data"
        data_path.mkdir(exist_ok=True)

        filename = f"{symbol.replace('/', '')}_{timeframe}.csv"
        filepath = data_path / filename

        if filepath.exists():
            data = pd.read_csv(filepath)
            # Ensure date columns are in datetime format
            data['date'] = pd.to_datetime(data['date'])
        else:
            data = fetch_ohlcv_data(exchange, symbol, data_start_date_str, hyperparameters['TEST_END_DATE'], timeframe)
            data.to_csv(filepath, index=False)

        # Prepare the data (calculate features) over the entire dataset
        prepared_data = prepare_data(data)

        # Convert 'date' and 'timestamp' columns to datetime for accurate splitting
        prepared_data['date'] = pd.to_datetime(prepared_data['date'])
        data['date'] = pd.to_datetime(data['date'])

        # Adjust data selection to include lookback data
        train_features = prepared_data[
            (prepared_data['date'] >= train_start_date_for_data) &
            (prepared_data['date'] <= train_end_date_dt)
        ].reset_index(drop=True)

        train_data = data[
            (data['date'] >= train_start_date_for_data) &
            (data['date'] <= train_end_date_dt)
        ].reset_index(drop=True)

        test_features = prepared_data[
            (prepared_data['date'] >= test_start_date_for_data) &
            (prepared_data['date'] <= test_end_date_dt)
        ].reset_index(drop=True)

        test_data = data[
            (data['date'] >= test_start_date_for_data) &
            (data['date'] <= test_end_date_dt)
        ].reset_index(drop=True)

        # Configuration for the environment with start dates
        config_train = {
            "data": train_data,
            "features": train_features,
            "start_date": train_start_date_dt
        }

        # Instantiate the training environment
        env = TradingEnv(config_train, device=device, start_date=train_start_date_dt)

        state_dim = env.state_dim
        action_dim = env.action_dim
        lookback = env.lookback

        # Instantiate the Discrete DDQN Agent
        discrete_agent = DiscreteDDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lookback=lookback,
            batch_size=hyperparameters['AGENT_BATCH_SIZE'],
            gamma=hyperparameters['AGENT_GAMMA'],
            lr=hyperparameters['AGENT_LR'],
            tau=hyperparameters['AGENT_TAU'],
            device=device,
            lora_r=hyperparameters['LORA_R'],
            lora_alpha=hyperparameters['LORA_ALPHA'],
            sigma_init=hyperparameters['SIGMA_INIT']
        )

        # Train the agent with resuming capability
        train_agents(env, discrete_agent, symbol, episodes=hyperparameters['NUM_EPISODES'], timeframe=timeframe, device=device, hyperparams=hyperparameters)

        # Generate signature for saving
        signature = generate_signature_from_hyperparams()

        # Prepare the test environment
        test_config = {
            "data": test_data,
            "features": test_features,
            "start_date": test_begin_date_dt
        }
        test_env = TradingEnv(test_config, device=device, start_date=test_begin_date_dt)

        # Initialize the agent for testing (load existing parameters)
        discrete_agent_test = DiscreteDDQNAgent(
            state_dim=test_env.state_dim,
            action_dim=test_env.action_dim,
            lookback=test_env.lookback,
            batch_size=hyperparameters['AGENT_BATCH_SIZE'],
            gamma=hyperparameters['AGENT_GAMMA'],
            lr=hyperparameters['AGENT_LR'],
            tau=hyperparameters['AGENT_TAU'],
            device=device,
            lora_r=hyperparameters['LORA_R'],
            lora_alpha=hyperparameters['LORA_ALPHA'],
            sigma_init=hyperparameters['SIGMA_INIT']
        )

        # Test and select episodes for RC
        test_and_select_rc(test_env, discrete_agent_test, symbol, timeframe=timeframe)

        # Save the experiment results
        save_experiment(symbol, timeframe, signature)

    print("All symbols have been processed.")

if __name__ == '__main__':
    main()
