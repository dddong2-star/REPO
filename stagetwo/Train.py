# 初始化
import datetime
import os
import random

import torch
from tqdm import tqdm

from stagetwo.Agent import REINFORCE, StateModel
from stagetwo.QNetwork import QNetwork, ReplayBuffer, compute_q_loss

import wandb

# # 初始化WandB
# wandb.init(project="relation-extraction-prompt-optimization", config={
#     "learning_rate": 1e-4,
#     "batch_size": 8,
#     "gamma": 0.95,
#     "epsilon": 0.5,
#     "target_update": 10,
#     "replay_buffer_size": 100
# })

# wandb.define_metric("custom_epoch")
# # define which metrics will be plotted against it
# wandb.define_metric("mean_P", step_metric="custom_epoch")
# wandb.define_metric("mean_R", step_metric="custom_epoch")
# wandb.define_metric("mean_F1", step_metric="custom_epoch")
# wandb.define_metric("mean_reward", step_metric="custom_epoch")
# wandb.define_metric("mean_similary", step_metric="custom_epoch")
# wandb.define_metric("current_q", step_metric="custom_epoch")
# wandb.define_metric("target_q", step_metric="custom_epoch")
# wandb.define_metric("Val/F1", step_metric="custom_epoch")
# wandb.define_metric("Train/Loss", step_metric="custom_epoch")


q_network = QNetwork()
target_network = QNetwork()
target_network.load_state_dict(q_network.state_dict())  # 初始化时同步
optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-4)
replay_buffer = ReplayBuffer(capacity=10000)
epsilon_max = 0.95  # 初始探索率
epsilon_min = 0.05  # 最低探索率
decay = 0.995  # 衰减率
batch_size = 8
target_update = 10
num_actions = 11
model_base_dir = "models_test"


def train_agent(train_dataset, val_dataset, test_dataset, num_episodes, agent, reward, state_model, mode):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 使用 os.path.join 进行安全拼接
    model_save_path = os.path.join(model_base_dir, current_time)
    # 初始化状态
    print("=======获取候选提示===========")
    initial_prompt = """从文本中提取涉及生产、供应、合作等关系，识别公司与其产品、服务或设备的生产商、供应商、合作伙伴等，并标注关系类型（如“生产”、“供应商”、“合作”等）。例如：（公司A，“生产”，产品B），（公司A，“供应商”，公司C）。"""
    initial_prompt = """根据下列新闻中的内容,总结新闻中的供应商关系,生产关系,构成关系。
                            以多个三元组的形式输出。没有此类关系则输出无。
                            关系描述如下:
                            1. 供应商关系:
                               如果句子中描述了A公司向B公司提供产品,可以抽取出(A, "供应商", B)。
                            2. 生产关系:
                               如果句子中描述了A公司生产某个产品P,可以抽取出 (A, "生产", P)。
                            3. 构成关系:
                               如果句子中描述了P产品作为X产品的构成部分,可以抽取出 (P, "构成", X)。

                             """

    """

                            供应关系满足一下句式:
                            1.A公司（为/向）B公司（提供/供应）P产品,则可以抽取出(A, "供应商", B),(A, "生产", P)。
                            2.A公司提供P产品,应用于B公司的产品,则可以抽取出(A, "供应商", B),(A, "生产", P)。
                            3.A公司是B公司的供应商,为B公司提供P产品,则可以抽取出(A, "供应商", B),(A, "生产", P)。
                            4.B公司是A公司的客户,则可以抽取出(A, "供应商", B)
                            5.A公司供货B公司,则可以抽取出(A, "供应商", B)
                            示例输入:趣睡科技董秘回复||| 财联社4月1日电,有投资者问,小米汽车已上市,请问公司目前跟小米汽车有合作吗,公司之前回复称有跟汽车厂商合作能透露是哪几家不？趣睡科技在互动平台表示,公司作为一家专注于自有品牌科技创新家居产品的互联网零售公司,公司积极开发车载家居产品,公司新开发的车载遮阳帘等产品已陆续上线销售,并与部分国内汽车厂商开展合作,小米汽车作为公司重要客户之一。
                            输出:（趣睡科技,"供应商",小米）,（小米,"生产",小米汽车）,（趣睡科技,"生产",车载遮阳帘）,（车载遮阳帘,"构成",小米汽车）

    """
    # 打印结果
    if mode == "train":
        dataset = train_dataset
    if mode == "test":
        dataset = test_dataset
    val_f1_history = []
    best_global_f1 = 0
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # model_save_path = model_save_path.join(f"{current_time}")

    # 创建文件夹（如果不存在）
    os.makedirs(model_save_path, exist_ok=True)
    best_global_prompt = None
    print("=======强化提示===============")
    # 总训练轮数
    with tqdm(total=num_episodes, desc="Training") as pbar:
        # 候选提示列表种的每一个提示
        prompt = initial_prompt
        for i_episode in range(num_episodes):
            # 训练
            state = state_model.get_state(prompt)
            epsilon = max(epsilon_min, epsilon_max * (decay ** i_episode))
            if random.random() < epsilon:
                action = random.randint(0, num_actions - 1)
            else:
                q_values = q_network(state)  # [batch,9]
                action = torch.argmax(q_values).item()
            # action=0
            # 执行动作
            if (i_episode + 1) % 5 == 0:
                new_prompt = REINFORCE().take_action(1, prompt)
            elif action == 8:
                new_prompt = REINFORCE().take_action(action, prompt, val_dataset)
            else:
                new_prompt = REINFORCE().take_action(action, prompt)
            print("==================action:" + str(action) + "=============================")
            print('\n')
            print(new_prompt)
            print("=======================================================")
            next_state = state_model.get_state(new_prompt)
            # 获取奖励
            text = train_dataset.source_texts
            gold_labels = train_dataset.re_labels
            # 这个奖励是平均的奖励reward
            rewards_tensor, rewards_log = reward.compute_reward(text, gold_labels, new_prompt, True)
            # 确保所有值都转换为 float 类型
            # wandb.log({"mean_P":rewards_log["mean_p"],"custom_epoch":i_episode})
            # wandb.log({"mean_R":rewards_log["mean_r"],"custom_epoch":i_episode})
            # wandb.log({"mean_F1":rewards_log["mean_f1"],"custom_epoch":i_episode})
            # wandb.log({"mean_reward":rewards_log["mean_reward"],"custom_epoch":i_episode})
            # wandb.log({"mean_similary":rewards_log["mean_similary"],"custom_epoch":i_episode})
            # 更新状态
            replay_buffer.push(state, action, rewards_tensor, next_state)
            prompt = new_prompt

            # 训练Q网络
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_q_loss(q_network, target_network, batch, i_episode)
                # 记录Loss
                # wandb.log({"Train/Loss": loss.item(),"custom_epoch": i_episode})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 更新目标网络
            if (i_episode + 1) % target_update == 0:
                print("========更新目标网络========")
                target_network.load_state_dict(q_network.state_dict())

            if i_episode % 5 == 0 and i_episode != 0:
                # 10轮评估一次
                print("=======评估=========")
                best_p, best_r, best_f1, best_prompt, best_actions = evaluate(q_network, state_model, reward,
                                                                              val_dataset, initial_prompt,
                                                                              model_save_path, i_episode)
                print("======评估结果=======")
                # 打印评估结果
                print(f"Epoch {i_episode}: F1={best_f1:.4f}")
                print(f"Best p:{best_p}")
                print(f"Best r:{best_r}")
                print(f"Best Prompt: {best_prompt}")
                print(f"Best action:" + str(best_actions))
                # wandb.log({
                #     "Val/best_f1": best_f1,"custom_epoch": i_episode})
                # wandb.log({
                #     "Val/avg_P": best_p, "custom_epoch": i_episode})
                # wandb.log({
                #     "Val/avg_R": best_r, "custom_epoch": i_episode})
                # val_f1_history.append(best_avg_f1)
                # 更新全局最优提示
                if best_f1 > best_global_f1:
                    best_global_f1 = best_f1
                    best_global_prompt = best_prompt
                    print(f"BEST_GLOBAL_F1:{best_global_prompt}")
                    # **保存最佳模型**
                    # torch.save(q_network.state_dict(), model_save_path.join("best_model.pth"))
                    # print(f"✅ 发现新最佳模型，已保存！F1: {best_global_f1:.4f}")

                if should_stop_training(i_episode, val_f1_history):
                    print("Training stopped.")
                    break

            pbar.set_postfix({'episode': '%d' % i_episode})
            pbar.update(1)
        print("结束啦")
        # 保存最优提示
    save_best_prompt(best_global_prompt, "best_prompt.txt")


def evaluate(q_network, state_model, reward, val_dataset, prompt, model_save_path, epoch):
    """
    评估当前Q网络生成的提示模板
    输入:
      - q_network: 当前的Q网络
      - val_dataset: 验证集
    输出:
      - avg_f1: 平均F1值
      - best_prompt: 最优提示模板
    """
    q_network.eval()  # 设置Q网络为评估模式
    best_prompt, best_f1 = None, 0
    best_p = 0
    best_r = 0
    stable_count = 0
    best_action_list = []
    action_list = []
    while stable_count <= 3:
        # 使用Q网络选择动作
        total_f1 = 0
        total_p = 0
        total_r = 0
        state = state_model.get_state(prompt)  # 初始状态
        with torch.no_grad():  # 评估时不计算梯度
            q_values = q_network(state)
            action = torch.argmax(q_values).item()
            # 选择最优动作
            action_list.append(action)
        # 执行动作
        if action == 8:
            new_prompt = REINFORCE().take_action(action, prompt, val_dataset)
        else:
            new_prompt = REINFORCE().take_action(action, prompt)
        for text, gold_triples in zip(val_dataset.source_texts, val_dataset.re_labels):
            # 在验证集上测试提示
            p, r, f1 = reward.evaluate_score(text, gold_triples, new_prompt)
            # 更新统计
            print(f"evluate_precision: {p:.2f}| "
                  f"evluate_recall: {r:.2f} | "
                  f"evluate_f1: {f1:.2f} | ")
            total_f1 += f1
            total_p += p
            total_r += r
        # 计算平均指标
        avg_f1 = total_f1 / len(val_dataset)
        avg_p = total_p / len(val_dataset)
        avg_r = total_r / len(val_dataset)
        print("avg_f1:" + str(avg_f1))
        # 记录评估指标
        if avg_f1 >= best_f1:
            best_p = avg_p
            best_r = avg_r
            best_f1 = avg_f1
            best_prompt = new_prompt
            best_action_list = action_list.copy()
            save_filename = f"model_epoch_{epoch}.pth"
            save_path = os.path.join(model_save_path, save_filename)
            torch.save(q_network.state_dict(), save_path)
            print(f"✅ 发现更优模型！已保存到 {save_path}")
        # if abs(avg_f1-previous_f1)<0.01:
        stable_count += 1
        # else:
        #     stable_count = 0  # 若波动较大，重置计数
        prompt = new_prompt
        # previous_f1=avg_f1
    return best_p, best_r, best_f1, best_prompt, best_action_list


def should_stop_training(epoch, val_f1_history, patience=5, threshold=0.85):
    """
    判断是否停止训练
    输入:
      - epoch: 当前训练轮数
      - val_f1_history: 验证集F1值历史记录
      - patience: 允许性能不提升的轮数
      - threshold: F1值阈值
    输出:
      - True/False: 是否停止训练
    """
    # 达到最大训练轮数
    if epoch >= 100:
        print("epoch >= 100")
        return True

    # 达到性能阈值
    # if val_f1_history[-1] >= threshold:
    #     print("val_f1_history[-1] >= threshold")
    #     return True

    # 性能收敛（F1值不再提升）
    if len(val_f1_history) > patience:
        recent_f1 = val_f1_history[-patience:]
        if max(recent_f1) - min(recent_f1) < 0.001:  # 最近patience轮F1值变化小于1%
            return True

    return False


def save_best_prompt(best_prompt, file_path):
    with open(file_path, 'w') as f:
        f.write(best_prompt)
    print("ok")
