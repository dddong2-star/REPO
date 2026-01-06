import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from collections import deque
import math


class CrossAttention(nn.Module):
    """交叉注意力层，用于融合结构化特征和描述文本"""

    def __init__(self, hidden_size, num_attention_heads=8, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 查询、键、值的线性变换
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # 输出投影
        self.output = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        # 线性变换并重塑形状
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax归一化
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 应用注意力权重
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 输出投影和残差连接
        output = self.output(context_layer)
        output = self.dropout(output)
        output = self.layer_norm(query_states + output)

        return output


class UnifiedEncoder(nn.Module):
    """统一编码器，处理结构化数据和描述文本的混合输入"""

    def __init__(self, model_name="bert-base-uncased"):
        super(UnifiedEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.cross_attention = CrossAttention(self.hidden_size)
        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask, token_type_ids=None, str_mask=None, desc_mask=None):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 如果未提供结构化和描述的mask，尝试根据特殊token分割
        if str_mask is None or desc_mask is None:
            # 这里简化处理，实际实现需要根据[STR]和[DESC]标记位置动态计算
            # 假设前半部分是结构化数据，后半部分是描述文本
            seq_len = sequence_output.size(1)
            mid_point = seq_len // 2

            str_mask = torch.zeros_like(attention_mask)
            str_mask[:, :mid_point] = attention_mask[:, :mid_point]

            desc_mask = torch.zeros_like(attention_mask)
            desc_mask[:, mid_point:] = attention_mask[:, mid_point:]

        # 提取结构化部分和描述部分的表示
        batch_size = sequence_output.size(0)

        # 创建掩码以提取各部分的表示
        str_mask = str_mask.unsqueeze(-1).expand(sequence_output.size())
        desc_mask = desc_mask.unsqueeze(-1).expand(sequence_output.size())

        # 应用掩码并进行平均池化
        str_tokens = sequence_output * str_mask
        str_sum = torch.sum(str_tokens, dim=1)
        str_count = torch.sum(str_mask[:, :, 0], dim=1, keepdim=True)
        str_representation = str_sum / (str_count + 1e-10)

        desc_tokens = sequence_output * desc_mask
        desc_sum = torch.sum(desc_tokens, dim=1)
        desc_count = torch.sum(desc_mask[:, :, 0], dim=1, keepdim=True)
        desc_representation = desc_sum / (desc_count + 1e-10)

        # 扩展为序列形式用于交叉注意力
        str_rep_expanded = str_representation.unsqueeze(1)  # [batch_size, 1, hidden_size]
        desc_rep_expanded = desc_representation.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 交叉注意力融合 - 使用结构化表示作为查询，描述表示作为键和值
        fused_representation = self.cross_attention(
            str_rep_expanded,
            desc_rep_expanded,
            desc_rep_expanded
        )

        # 最终表示
        output = self.activation(self.pooler(fused_representation.squeeze(1)))
        return output


class MomentumContrastive(nn.Module):
    """基于MoCo的动量对比学习框架"""

    def __init__(
            self,
            model_name="bert-base-uncased",
            dim=768,
            queue_size=4096,
            momentum=0.999,
            temperature=0.07,
    ):
        super(MomentumContrastive, self).__init__()

        # 初始化在线编码器和动量编码器
        self.encoder_q = UnifiedEncoder(model_name)
        self.encoder_k = UnifiedEncoder(model_name)

        # 初始化动量编码器参数，并设置为不需要梯度
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 投影头 - 可选，若需要将表示投影到不同维度
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新策略，同步更新动量编码器参数"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新负样本队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # 替换队列中的样本
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 处理队列边界情况
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        # 更新指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, query_input, key_input=None):
        """前向传播，计算对比损失

        Args:
            query_input: 查询样本的输入 (input_ids, attention_mask, token_type_ids)
            key_input: 正样本的输入，如果为None则使用查询样本作为正样本

        Returns:
            loss: 对比损失
            q: 查询表示
            k: 正样本表示
        """
        if key_input is None:
            key_input = query_input

        q_input_ids, q_attention_mask = query_input['input_ids'], query_input['attention_mask']
        k_input_ids, k_attention_mask = key_input['input_ids'], key_input['attention_mask']

        # 计算查询表示
        q = self.encoder_q(q_input_ids, q_attention_mask)
        q = self.projector(q)
        q = F.normalize(q, dim=1)

        # 计算正样本表示，不需要梯度
        with torch.no_grad():
            # 更新动量编码器
            self._momentum_update_key_encoder()

            k = self.encoder_k(k_input_ids, k_attention_mask)
            k = self.projector(k)
            k = F.normalize(k, dim=1)

        # 计算logits
        # 正样本得分
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # 负样本得分
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: [batch_size, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1)

        # 应用温度系数
        logits /= self.temperature

        # 标签：第一个是正样本
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        # 更新队列
        self._dequeue_and_enqueue(k)

        return loss, q, k


class UnifiedEntityLinkingModel:
    """统一实体链接模型的完整封装"""

    def __init__(
            self,
            model_name="bert-base-uncased",
            queue_size=4096,
            momentum=0.999,
            temperature=0.07,
            learning_rate=2e-5,
            weight_decay=0.01
    ):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # 添加特殊token
        special_tokens = {"additional_special_tokens": ["[STR]", "[DESC]"]}
        self.tokenizer.add_special_tokens(special_tokens)

        # 初始化模型
        self.model = MomentumContrastive(
            model_name=model_name,
            queue_size=queue_size,
            momentum=momentum,
            temperature=temperature
        )

        # 调整BERT以适应新token
        self.model.encoder_q.bert.resize_token_embeddings(len(self.tokenizer))
        self.model.encoder_k.bert.resize_token_embeddings(len(self.tokenizer))

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def prepare_input(self, str_text, desc_text, max_length=512):
        """准备模型输入，将结构化文本和描述文本整合为单一序列"""
        input_text = f"[STR] {str_text} [DESC] {desc_text}"
        encoded = self.tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 计算结构化部分和描述部分的mask
        input_ids = encoded["input_ids"][0]
        str_start = (input_ids == self.tokenizer.convert_tokens_to_ids("[STR]")).nonzero().item()
        desc_start = (input_ids == self.tokenizer.convert_tokens_to_ids("[DESC]")).nonzero().item()

        str_mask = torch.zeros_like(encoded["attention_mask"])
        str_mask[:, str_start:desc_start] = encoded["attention_mask"][:, str_start:desc_start]

        desc_mask = torch.zeros_like(encoded["attention_mask"])
        desc_mask[:, desc_start:] = encoded["attention_mask"][:, desc_start:]

        encoded["str_mask"] = str_mask
        encoded["desc_mask"] = desc_mask

        return encoded

    def train_step(self, query_batch, key_batch=None):
        """执行一步训练"""
        self.model.train()

        # 如果没有提供正样本，则查询样本自身作为正样本
        if key_batch is None:
            key_batch = query_batch

        # 将输入移动到GPU（如果可用）
        device = next(self.model.parameters()).device
        for k, v in query_batch.items():
            query_batch[k] = v.to(device)
        for k, v in key_batch.items():
            key_batch[k] = v.to(device)

        # 前向传播
        loss, _, _ = self.model(query_batch, key_batch)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def encode_entity(self, str_text, desc_text):
        """对单个实体进行编码，返回统一表示"""
        self.model.eval()

        # 准备输入
        inputs = self.prepare_input(str_text, desc_text)

        # 将输入移动到GPU（如果可用）
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 使用在线编码器进行编码
        with torch.no_grad():
            # 获取实体表示
            entity_repr = self.model.encoder_q(
                inputs["input_ids"],
                inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                str_mask=inputs.get("str_mask"),
                desc_mask=inputs.get("desc_mask")
            )
            # 应用投影头
            entity_repr = self.model.projector(entity_repr)
            # 归一化
            entity_repr = F.normalize(entity_repr, dim=1)

        return entity_repr.cpu().numpy()

    def save_model(self, path):
        """保存模型参数和tokenizer"""
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f"{path}/model.pt")

        # 保存tokenizer
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        """加载模型参数和tokenizer"""
        # 加载模型参数
        checkpoint = torch.load(f"{path}/model.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(path)


# 使用示例
def example_usage():
    # 初始化模型
    model = UnifiedEntityLinkingModel(model_name="bert-base-uncased")

    # 示例数据
    entity1_str = "类型: 公司; 成立时间: 1976年; 市值: 2.7万亿美元"
    entity1_desc = "苹果公司是一家专注于消费电子、软件和在线服务的跨国公司。"

    entity2_str = "类型: 公司; 成立时间: 1998年; 市值: 1.5万亿美元"
    entity2_desc = "谷歌是一家专注于互联网搜索、在线广告和云计算的科技公司。"

    # 准备批次训练数据
    query_batch = model.prepare_input(entity1_str, entity1_desc)
    key_batch = model.prepare_input(entity2_str, entity2_desc)

    # 训练示例
    loss = model.train_step(query_batch, key_batch)
    print(f"Training loss: {loss}")

    # 编码实体示例
    entity_repr = model.encode_entity(entity1_str, entity1_desc)
    print(f"Entity representation shape: {entity_repr.shape}")

    # 保存和加载模型
    # model.save_model("./saved_model")
    # model.load_model("./saved_model")


if __name__ == "__main__":
    example_usage()