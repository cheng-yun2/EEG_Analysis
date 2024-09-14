import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from jedi.debug import print_to_stdout
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 读取数据
all_data = pd.read_csv('./X_train/X_train').to_numpy()
# 假设你想要70%的数据用作训练，30%的数据用作测试
X_train, X_test = train_test_split(all_data, test_size=0.3, random_state=42)
y_train = X_train[:,-1]
y_test = X_test[:,-1]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# 数据标准化
scaler = StandardScaler()
print('-'*50)
X_train = scaler.fit_transform(X_train[:,:-1])
X_test = scaler.transform(X_test[:,:-1])
print('-'*50)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 定义数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 应用dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)  # 应用dropout
        x = self.fc3(x)
        return x



# 设置模型参数
input_size = X_train.shape[1]
hidden_size = 64  # 隐藏层神经元数量
num_classes = 9  # 类别数（多分类）
learning_rate = 0.001
num_epochs = 100

# 初始化模型并将其移动到GPU
model = MLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类任务用 CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    # 保存最好的模型
    if epoch_acc > best_accuracy:
        best_accuracy = epoch_acc
        torch.save(model.state_dict(), f'model_best.pth')
        print("Saved Best Model")

# 评估模型
model.eval()  # 设置为评估模式
correct = 0
total = 0

with torch.no_grad():  # 禁用梯度计算，节省计算和内存使用
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 确保数据也在正确的设备上
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算准确率
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

