import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# 数据预处理类
class PowerDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        """清洗数据中可能的格式问题"""
        import re
        
        # 处理可能连在一起的数值列
        numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                          'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
                          'Sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
        
        for col in numeric_columns:
            if col in df.columns:
                # 如果是字符串类型，尝试修复
                if df[col].dtype == 'object':
                    print(f"Cleaning column: {col}")
                    # 处理可能连在一起的数值
                    new_values = []
                    for i, val in enumerate(df[col]):
                        if pd.isna(val):
                            new_values.append(np.nan)
                        else:
                            val_str = str(val)
                            # 处理特殊情况：如 '0.0.1' -> '0.0'
                            if '..' in val_str:
                                val_str = val_str.split('..')[0]
                            elif val_str.count('.') > 1:
                                # 如果有多个小数点，取第一个有效的数字部分
                                parts = val_str.split('.')
                                if len(parts) >= 2:
                                    val_str = parts[0] + '.' + parts[1]
                            
                            # 如果字符串很长，可能是连在一起的数值
                            if len(val_str) > 20:
                                # 尝试分割连续的数值
                                # 假设数值格式为 XXX.XXX
                                numbers = re.findall(r'\d+\.\d+', val_str)
                                if numbers:
                                    # 取第一个数值
                                    try:
                                        new_values.append(float(numbers[0]))
                                    except:
                                        new_values.append(np.nan)
                                else:
                                    # 如果没有找到小数点格式，尝试整数
                                    numbers = re.findall(r'\d+', val_str)
                                    if numbers:
                                        try:
                                            new_values.append(float(numbers[0]))
                                        except:
                                            new_values.append(np.nan)
                                    else:
                                        new_values.append(np.nan)
                            else:
                                try:
                                    new_values.append(float(val_str))
                                except:
                                    new_values.append(np.nan)
                    
                    df[col] = new_values
                
                # 确保转换为数值类型
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    def process_data(self, df):
        """按天汇总数据"""
        # 首先清洗数据
        df = self.clean_data(df)
        
        # 确保DateTime列存在且格式正确
        if 'DateTime' not in df.columns:
            raise ValueError("DateTime column not found in data")
            
        # 转换日期时间格式
        try:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
        except Exception as e:
            print(f"Error converting DateTime: {e}")
            # 尝试其他可能的格式
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
            except:
                df['DateTime'] = pd.to_datetime(df['DateTime'], infer_datetime_format=True)
        
        df.set_index('DateTime', inplace=True)
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 按天汇总
        daily_data = df.resample('D').agg({
            'Global_active_power': 'sum',
            'Global_reactive_power': 'sum', 
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum',
            'RR': 'first',
            'NBJRR1': 'first',
            'NBJRR5': 'first',
            'NBJRR10': 'first',
            'NBJBROU': 'first'
        })
        
        # 计算sub_metering_remainder
        daily_data['sub_metering_remainder'] = (
            daily_data['Global_active_power'] * 1000 / 60 - 
            daily_data['Sub_metering_1'] - 
            daily_data['Sub_metering_2'] - 
            daily_data['Sub_metering_3']
        )
        
        # 再次处理缺失值
        daily_data = daily_data.fillna(method='ffill').fillna(method='bfill')
        
        # 如果还有缺失值，用0填充
        daily_data = daily_data.fillna(0)
        
        return daily_data
    
    def create_sequences(self, data, seq_length, pred_length):
        """创建序列数据"""
        X, y = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length:i + seq_length + pred_length, 0])  # 预测Global_active_power
        return np.array(X), np.array(y)

# 数据集类
class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Enhanced ConvInformer模型
class EnhancedConvInformer(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_layers,
        output_size,
        conv_kernels=(3,5,7),
        dilations=(1,2,4),
        ffn_dim=None,
        dropout=0.2
    ):
        super(EnhancedConvInformer, self).__init__()
        if ffn_dim is None:
            ffn_dim = d_model * 4

        # 输入映射与位置编码
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        # 多头注意力层列表
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        # 多尺度卷积模块列表
        self.multi_conv_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=k, dilation=d, padding=((k-1)//2)*d)
                for k, d in zip(conv_kernels, dilations)
            ])
            for _ in range(num_layers)
        ])

        # 通道注意力门控
        self.channel_gate_fc1 = nn.ModuleList([
            nn.Linear(d_model, d_model // 4)
            for _ in range(num_layers)
        ])
        self.channel_gate_fc2 = nn.ModuleList([
            nn.Linear(d_model // 4, d_model)
            for _ in range(num_layers)
        ])

        # 前馈网络
        self.ffn_fc1 = nn.ModuleList([
            nn.Linear(d_model, ffn_dim)
            for _ in range(num_layers)
        ])
        self.ffn_fc2 = nn.ModuleList([
            nn.Linear(ffn_dim, d_model)
            for _ in range(num_layers)
        ])

        # 层归一化
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # 输出投影
        self.output_fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        for i in range(len(self.attention_layers)):
            # 自注意力子层
            attn = self.attention_layers[i]
            attn_out, _ = attn(x, x, x)
            x1 = self.norm_layers[i](x + self.dropout(attn_out))

            # 多尺度卷积
            convs = self.multi_conv_layers[i]
            conv_input = x1.transpose(1, 2)  # (batch, d_model, seq_len)
            conv_outputs = []
            for conv in convs:
                conv_o = conv(conv_input)
                conv_outputs.append(conv_o)
            multi_conv = torch.stack(conv_outputs, dim=-1).sum(-1)
            conv_proj = multi_conv.transpose(1, 2)  # (batch, seq_len, d_model)
            x2 = self.norm_layers[i](x1 + self.dropout(conv_proj))

            # 通道注意力门控
            gate = conv_proj.mean(dim=1)  # (batch, d_model)
            gate = F.relu(self.channel_gate_fc1[i](gate))
            gate = torch.sigmoid(self.channel_gate_fc2[i](gate)).unsqueeze(1)  # (batch, 1, d_model)
            x3 = x1 * gate + x2 * (1 - gate)

            # 前馈网络
            f = F.relu(self.ffn_fc1[i](x3))
            f = self.ffn_fc2[i](self.dropout(f))
            x = self.norm_layers[i](x3 + self.dropout(f))

        # 输出层
        out = x[:, -1, :]  # 取最后时间步
        out = self.output_fc(self.dropout(out))
        return out


# 训练函数
def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# 预测函数
def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
    return np.array(predictions)

# 评估函数
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

def load_and_examine_data(file_path):
    """加载并检查数据格式"""
    try:
        # 首先尝试正常加载
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # 检查是否是测试数据格式问题（第一行数据被当作列名）
        if 'DateTime' not in df.columns and len(df.columns) == 13:
            print("Detected test data format issue, fixing...")
            
            # 重新读取，不使用第一行作为列名
            df = pd.read_csv(file_path, header=None)
            
            # 设置正确的列名
            df.columns = ['DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage', 
                         'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 
                         'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
            
            print(f"Fixed shape: {df.shape}")
            print(f"Fixed columns: {df.columns.tolist()}")
        
        # 检查数据类型
        print("\nData types:")
        print(df.dtypes)
        
        # 检查前几行数据
        print("\nFirst few rows:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# 主函数
def main():
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    train_df = load_and_examine_data('train.csv')
    test_df = load_and_examine_data('test.csv')
    
    if train_df is None or test_df is None:
        print("Failed to load data files.")
        return
    
    # 数据预处理
    processor = PowerDataProcessor()
    
    try:
        train_processed = processor.process_data(train_df.copy())
        test_processed = processor.process_data(test_df.copy())
        
        print(f"Training data shape: {train_processed.shape}")
        print(f"Test data shape: {test_processed.shape}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    # 特征标准化
    train_scaled = processor.scaler.fit_transform(train_processed)
    test_scaled = processor.scaler.transform(test_processed)
    
    # 实验配置
    seq_length = 90
    pred_lengths = [90, 365]  # 短期和长期预测
    input_size = train_scaled.shape[1]
    num_runs = 5
    
    results = {}
    
    for pred_length in pred_lengths:
        print(f"\n=== Predicting {pred_length} days ===")
        
        # 创建序列数据
        X_train, y_train = processor.create_sequences(train_scaled, seq_length, pred_length)
        X_test, y_test = processor.create_sequences(test_scaled, seq_length, pred_length)
        
        # 分割训练和验证集
        val_size = int(0.2 * len(X_train))
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # 模型配置
        models_config = {
            'LSTM': {
                'model': LSTMModel,
                'params': {
                    'input_size': input_size,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'output_size': pred_length,
                    'dropout': 0.2
                }
            },
            'Transformer': {
                'model': TransformerModel,
                'params': {
                    'input_size': input_size,
                    'd_model': 128,
                    'nhead': 8,
                    'num_layers': 4,
                    'output_size': pred_length,
                    'dropout': 0.2
                }
            },
            'Informer': {
                'model': EnhancedConvInformer,
                'params': {
                    'input_size': input_size,
                    'd_model': 128,
                    'nhead': 8,
                    'num_layers': 3,
                    'output_size': pred_length,
                    'dropout': 0.2
                }
            }
        }
        
        results[pred_length] = {}
        
        for model_name, config in models_config.items():
            print(f"\n--- Training {model_name} ---")
            
            mse_scores = []
            mae_scores = []
            
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}")
                set_seed(42 + run)
                
                # 创建数据加载器
                train_dataset = PowerDataset(X_train, y_train)
                val_dataset = PowerDataset(X_val, y_val)
                test_dataset = PowerDataset(X_test, y_test)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # 初始化模型
                model = config['model'](**config['params'])
                
                # 训练模型
                train_losses, val_losses = train_model(
                    model, train_loader, val_loader, 
                    epochs=50, learning_rate=0.001, device=device
                )
                
                # 预测
                predictions = predict(model, test_loader, device)
                
                # 评估
                mse, mae = evaluate_model(y_test, predictions)
                mse_scores.append(mse)
                mae_scores.append(mae)
                
                print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")
            
            # 计算平均结果和标准差
            avg_mse = np.mean(mse_scores)
            std_mse = np.std(mse_scores)
            avg_mae = np.mean(mae_scores)
            std_mae = np.std(mae_scores)
            
            results[pred_length][model_name] = {
                'MSE': {'mean': avg_mse, 'std': std_mse},
                'MAE': {'mean': avg_mae, 'std': std_mae}
            }
            
            print(f"{model_name} Results:")
            print(f"  MSE: {avg_mse:.4f} ± {std_mse:.4f}")
            print(f"  MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    
    # 打印最终结果
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    for pred_length in pred_lengths:
        print(f"\n{pred_length}-day prediction:")
        print("-" * 30)
        for model_name in ['LSTM', 'Transformer', 'Informer']:
            result = results[pred_length][model_name]
            print(f"{model_name}:")
            print(f"  MSE: {result['MSE']['mean']:.4f} ± {result['MSE']['std']:.4f}")
            print(f"  MAE: {result['MAE']['mean']:.4f} ± {result['MAE']['std']:.4f}")

if __name__ == "__main__":
    main()