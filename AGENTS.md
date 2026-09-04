# AGENTS.md

## Language

- 默认使用中文回复，并称呼用户为 `nullptr`。
- English terms must include Chinese meaning on first use, for example GitHub（代码托管平台）, Git（版本控制系统）, branch（分支）, commit（提交）, and push（推送）。
- 不确定时必须明确说明不确定点，不要编造不存在的文件、接口、命令或结果。
- 生成的各种数据文件，png，csv，md等，尽量使用中文描述，如是英语专业名称，旁边需要附上中文解释。md文件也尽量用中文命名。
- 生成的md文件，log文件等命名前加上gpt-前缀，便于区分人工和AI生成的文件。
- md里的数学公式要渲染正确，不要使用\[
P_{\mathrm{cost}}=P_{\mathrm{el}}+P_{\mathrm{mech}}.
\],$r$这种无法渲染成功的格式。
- 对于分步骤的任务，在每次会话结束后应主动告诉我下一步要怎么做。成功完成任务时，应主动告诉我当前任务完成。

## Environment

- 本机是 Ubuntu 22.04服务器，无GUI环境。gpu0和gpu1是A100，gpu2-5是V100。我使用VSCode的SSH拓展远程连接。
- gpu长训练（例如3000次it）必须在tmux或GNU Screen等可断线恢复的终端会话中运行，短训练冒烟测试等无所谓。gpu训练任务只要能在gpu上正常启动即可，无需担心抢占他人任务。
- 给出的命令行尽量不要用export，选定显卡应该在命令行中直接指定CUDA_VISIBLE_DEVICES，而不是在~/.bashrc或~/.zshrc中设置。
- 命令行有修改变动，应该跟我说明改动会影响什么，特别是会不会影响训练出来的模型权重。
- 正式评估不强制使用 tmux 或 GNU Screen，可以在普通终端直接运行。
- 无真实机器人等设备，目前只能做仿真。录制仿真中的运动视频只能使用gpu0和gpu1。
- 对于长命令行使用单行命令行，不要使用换行符和反斜杠。

## GitHub Network

- 2026-08-20确认：服务器到GitHub的国际出口链路不稳定。直连HTTPS可以偶尔完成只读查询，但`git push`的数据上传请求会出现`HTTP 408 Request Timeout`；标准SSH 22端口不可用。这不是仓库权限错误，也不能据此判断GitHub全局故障。
- 当前Git配置通过`http://127.0.0.1:27897`访问GitHub。端口27897来自用户Windows电脑建立的SSH隧道；端口17897用于SSH连接本身。
- 任何`git fetch`、`git pull`、`git push`或GitHub API写操作前，先在服务器执行`ss -ltn | rg '127\.0\.0\.1:27897'`。如果没有监听，不要反复重试写操作，应请用户在Windows PowerShell运行`ssh -N -T xjy-clash-tunnel`并保持窗口运行；用户也提供了`ssh -N -T xjy-v2rayn-tunnel`作为备选隧道命令。
- 上述`ssh -N -T ...`命令必须在用户的Windows PowerShell执行，不是在Ubuntu服务器执行；相关SSH别名定义在用户本地配置中。
- 隧道可用时使用仓库的正常Git命令，让既有代理配置生效。不要在push时添加`-c http.https://github.com.proxy=`来绕过代理；服务器直连上传已被验证会超时。
- push输出若同时出现`HTTP 408`、`unexpected disconnect`和`Everything up-to-date`，不得把最后一行解释为成功。必须使用`git ls-remote --heads origin '<分支名>'`核对远端分支，并比较远端与本地完整commit哈希。
- 服务器当前没有可用于`git@github.com`的SSH公钥认证；不要擅自把remote改为SSH，也不要改写用户Git代理或凭据配置。
- 网络诊断不得输出访问令牌、Authorization头或`~/.config/gh/hosts.yml`内容。如确需curl/Git跟踪，必须启用脱敏并只报告HTTP状态、阶段和耗时。

## rule
- 本工程以 [LeggedGym](https://github.com/leggedrobotics/legged_gym/tree/8fa29acc6fd1910c3d9659eef6310bdd301cde0a)、Isaac Gym Preview 4 和 rsl_rl v1.0.2 为基础，按照[论文 v2](https://arxiv.org/html/2509.06342v2)、作者回复及 pace_data 重建 ANYmal D 混合地形 PPO 环境

- 1. **Stage 0：纯 PACE 环境链路验证。** 先只搭好 ANYmal + LeggedGym / Isaac Gym，把 PACE 的机器人参数、PD、识别动力学、terrain、command、observation 和 reward 实现好。这时候甚至先不关心能耗优化，目标只有一个：机器人能正常站立、走路、跟踪速度，不出现明显实现错误。同时单独验证 `0.0192 * Στ²`、机械功率、总能耗的计算量级和符号是否合理。 
2. **Stage 1：Task-only PPO。** 去掉 energy penalty，只保留 PACE 的任务 reward、collision、FTD 等结构，训练一个稳定的 ANYmal locomotion policy。这个阶段非常重要，因为它回答：“我的 LeggedGym 环境本身能不能学会走路？”如果 Task-only 都学不起来，就绝对不要碰 PPO-Lagrangian。 
3. **Stage 2：PACE-Curriculum baseline。** 在完全相同的环境和 PPO 设置上加入 PACE 原论文的 energy reward 和 exponential curriculum，包括 500 iteration half-life。目标是证明：你的实现能够复现 PACE 所描述的基本现象——加入能耗惩罚后，任务仍然能学会，同时能耗发生合理下降。到这里为止，**完全不要加入 ECO**。 
4. **Stage 3：只加入一个 ECO energy constraint。** 保留 Stage 2 的所有机器人、环境、task reward、FTD curriculum、PPO 主体参数，只删除
   ```math
   \kappa c_e r_e
   ```
   这一项，然后增加 cost critic、Lagrange multiplier 和
   ```math
   J_C(\pi)\le B
   ```
   。这才是第一版真正的 **PACE-ECO**。此时不要加入 mirror/symmetry constraint。
5. **Stage 4：预算和 PPO-Lagrangian 稳定性实验。** 用 Task-only 重新标定 Isaac Gym 环境自己的 `B_{\rm ref}`，例如先测试 `B_{90},B_{80},B_{70}` 之类的约束强度，观察任务性能—能耗 Pareto 关系，同时检查 `\lambda` 是否发散、cost critic 是否能学准、constraint violation 是否逐渐收敛。 
6. **Stage 5：只有出现明确步态问题时再加 symmetry。** 比如发现 Energy-only 策略长期偏向某几条腿、左右/前后负载明显不均，或者为了省能耗学出异常 gait，才增加 symmetry constraint，并把它作为单独消融：

   Energy-onlyvsEnergy+Symmetry.

- 减少冻结，HASH等防御性编程的过度使用。不要因为防御性编程忘记项目主旨。
