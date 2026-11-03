# VITRA H1 12D 逆解码实验（2026-09-08）

## 结论与范围

已实现显式 opt-in 的非线性 `Wuji20 → H1 independent12` 配对解码器，保持 0902 MANO45 checkpoint、statistics、normalizer、16-step chunk 和 EE 腕部链路不变。未修改 Eval Web、scheduler、public loop、白名单、DexBench 配置或任何质量阈值。旧 geometric/linear-fallback 默认行为保留。

这不是丢失信息的精确逆映射：原训练预处理仅读取 6 个 H1 角，另外 6 个真实独立角被 Spark mimic 取代。新解码器对它们做条件估计，不能保证恢复任意真实 12D 姿态。若要求保留全部独立关节信息，必须更改训练表示并重新训练/微调手部输出头。

## 已修改的线上代码

远端前缀：`/personal/xiangpc/0813_Xpolicylab_bench/VITRA/XPolicyLab/policy/VITRA/`。

- `model.py`：仅增加 H1 分支下的显式 `h1_paired12_path`/`h1_paired12_sha256` 配置（也接受对应 `VITRA_H1_PAIRED12_*` 环境变量）。未配置时不改变原行为。
- 新增 `h1_paired12_inverse.py`：读取 hash 锁定的数值树模型；验证 stats/config/manifest/checkpoint 路径与源文件 hash；输出真实顺序 12D，不做 mimic、clip、padding、truncation 或静默 fallback。保留原 0.08 m 几何门，并明确其不能证明 12D 重建正确。
- 新增 `artifacts/egovla_paired12_20260908/`：原模型源码备份、拟合脚本、CPU 测试、数值森林、sidecar、报告与独立运行目录。未覆盖旧 artifact。

|文件/对象|修改前 SHA256|修改后 SHA256|
|---|---|---|
|model.py|a19071fad9c2534e6a84f0397e75e48528ced55fd43f0370675f09a537b6648b|8dae7bf36aa38364f2274d3db4ca4ae874e6f72d02223513737299fc676dde37|
|h1_paired12_inverse.py|新增|6d89a5b2c333f69380bebfc7a06ad1478f1026b9d1e693e6d8a7dead03f08611|
|paired12_0908_v1.json|新增|b1f734d1e64280f0226d62c3fb12ada8c60946bc23a3939e4b080b91589b1f4b|

原 `h1_inspire_retarget.py` 未改，SHA `036f672c19f70290ec4d7c0e0081d905f93b7b034cb1c93943665fa1816db98f`。已有 `points[:,0,:]=0` 修正保持，不将它再次报告为新根因。

## Checkpoint provenance

Run 相对 policy 路径：
`checkpoints/egovla_humanoid_mano-20260902-tianji_marvin_wuji-ee-42/2026-09-02-egovla_humanoid_mano-20260902_TB64_B8_bf16True/`。

|对象|SHA256（本次未改）|
|---|---|
|checkpoints/epoch=10-step=80000.ckpt/weights.pt|6833a0b06137a5b330193ad3b5cde6834bdf00ea3e13752a89a19656ad270803|
|config.json|755469f035d417fc3c3582552518c17db366ed48067e07be8a9e19817415f378|
|0902 teledata_statistics.json|8c0170503168fba85850a0842c35081f00e9f9084b06d678381a73742b107003|
|0902 spark0_manifest.json|ba6aa23212581ba1b426853fe4ac13129bb13e9d4c9cf754be4b868f89f26ee2|
|mano45_to_wuji20_linear_0902_v1.json|f0a760427d6fb7fd7aa9c7d63aea8666c67f4f7344c25e93f8d7f1adf05caba5|
|真实 H1 USD|c66589fffb2520cf51403ff5fd62569a18f4994b323ba53414d49acc7f8ec75d|

Statistics 来源：`/personal/xiangpc/0813_Xpolicylab_bench/VITRA/data/egovla_humanoid_sim_mano_20260902/teledata_statistics.json`。未使用旧 1-byte statistics。

内部每手 state/action 仍为 61/51；双手 action chunk 为 `(16,102)`（原生数值表示），下游 EE 接口每手腕部 pose7 + 手12，合计 `(16,38)`。真实手状态输入每手12，未冒充标准 joint 模型。

## CPU 证据

固定 `random_state=20260908`，ExtraTrees 64 棵，深度上限22，叶节点至少8个样本；未根据留出集调参。按 manifest 每任务前40个 episode 取样，共480个 episode，不能视作全1903 episode验收。

`Flip-Mug`、`Pour-Balls` 整任务留出；其他任务 episode编号 `%5==0` 整 episode 留出。训练/episode留出/task留出原始行数分别 67,893 / 18,747 / 10,478（每手）。无相邻帧随机切分。

原始 qpos 包含超出当前 USD nominal limits 的记录。第一版直接拟合会产生越界输出，因此未接入。第二版训练仅保留原始12维全部位于真实限位内的行，左53,400行、右51,475行；分别排除14,493/16,418行，未修改标签。所有留出行仍保留计分，包括越界标签。树叶均值自然有界，不靠运行时裁剪。

以下为每侧/每类留出集固定均匀取96行，同一输入比较；旧解码96/96均接受，未排除旧失败以美化结果：

|留出|旧12D RMS(rad)|新12D RMS(rad)|旧指尖RMS(mm)|新指尖RMS(mm)|
|---|---:|---:|---:|---:|
|左/episode|0.17644|0.04438|16.52|4.11|
|左/整任务|0.16417|0.06231|17.32|6.90|
|右/episode|0.21077|0.08151|19.19|5.52|
|右/整任务|0.25605|0.17050|20.02|11.83|

完整留出集新12D RMS：左 episode/task `0.07146/0.05602 rad`，右 `0.08697/0.16835 rad`。右手跨任务性能明显弱，不能以平均几何误差代替语义正确性。

原18样本：旧保留6角平均逐样本RMS `0.16678866 rad`，新 `0.05636579 rad`。以原 MANO affine 原始预测输入新解码器，全12角平均逐样本RMS `0.07489349 rad`；这是离线 raw-affine 结果，不冒充 public MANO safety/fallback 或神经模型完整输出。

四个合成 oracle 已复现。新闭手 retained6 RMS 左/右为 `0.08412/0.11068 rad`（旧 `0.36694/0.38211`）。中等姿态右手 retained6 `0.10953` 略差于旧 `0.10794`；其指尖和拇指对掌向量误差仍改善。没有声称每个关节、每个姿态全部改善。

数值森林与 sklearn 相同输入最大差 `2.22e-16 rad`；非法形状与 NaN 输入会显式报错。左右各96个整任务留出输入通过原80mm门，最大几何RMS `30.19/31.97 mm`。

## 真实本体、腕部与时间轴

直接读取当前 H1 USD 的 Revolute/FixedJoint，长度单位1米。各关节旋转轴为 joint-local Z，左右符号由各自 localRot0/localRot1 决定，不能把局部轴直接当世界轴。提取完整关节框架到 `h1_usd_joints.json`。

H1顺序：index近端/中间、middle近端/中间、pinky近端/中间、ring近端/中间、thumb yaw/pitch/intermediate/distal。关节限位(rad)：前8维 `[0,1.7]`，thumb yaw `[-0.1,1.3]`，其余 `[0,0.5]`、`[0,0.8]`、`[0,1.2]`。`h1_inspire.py` 对全部12维施加独立驱动。

用 USD 关节框架 FK、原始12D qpos、录制 EE xyz+wxyz 重建录制指尖，3任务×3帧×2手最大误差 `1.821e-7 m`，验证了索引、左右手映射和腕部 quaternion 顺序。没有据 Link7 名称更改 mount/camera。

实际 MANO fit→runtime inverse 的四个已知腕部姿态测试，平移误差0，旋转误差左 `0.01615 rad` / 右 `0.01589 rad`。四次均被现有 MANO velocity gate 转入已有 geometric fallback，报告中明确记录，未隐藏。

数据按每个 processed HDF5 内 provenance source_path 找到 raw，严格校验帧数和 `action_semantics=next_state_truncated:1`，使用 `raw qpos[kept_start+t+1]`，只取 manifest 定义的 training_transitions，没有 `min(len)` 静默截断。记录低通参数0.35、IK stride2与插值历史；当前原始 qpos 的来源代码并不对手角直接做 EMA，因此不能擅自再平滑标签。模型原有反归一化先于 MANO→Wuji→H1，原有16步chunk与stride2状态缓存保持不变。

## 验收状态

CPU、Python编译、三个现有启动脚本的 `bash -n`、真实 Isaac Python 的 PyYAML 导入、env-client dry-run 均已完成。

32步真实 Isaac debug 已尝试。服务器成功加载目标 weights、MANO45、paired12 sidecar、statistics 和 camera profile，并在 `localhost:43225` 监听；等待脚本确认 CONNECTED（服务就绪检查，不等于仿真客户端完成策略请求）。Isaac env-client 在进入仿真前返回：

`Isaac Sim requires explicit NVIDIA Omniverse EULA acceptance; after reviewing the license, export ACCEPT_EULA=Y before runtime evaluation`

首次没有代用户接受许可，退出后清理测试服务器。随后用户明确同意 EULA，才为本次运行设置 `ACCEPT_EULA=Y`；未使用 `--force`。

重新运行的32步真实 debug 已完成：`status=completed`，completed_episode_count=1，episode `error=null`；`episodes.jsonl`、`result.json`、`metrics.json` 和 MP4 均生成。MP4为1280×720、32帧、495,736字节，时长2.134秒，通过 ffprobe 和完整 ffmpeg 解码。32步限制触发正常 truncation，success_rate=0，不能当作正式任务成功率。

之后才启动 `official/single` 单回合，使用任务默认400步，不继承debug的32步限制。正式回合已完成：

- evaluation_id：`egovla-VITRA-pour_balls-9ef66c2147`。
- `status=completed`、completed_episode_count=1、episode `error=null`、steps=400。
- **success_rate=0.0（0%）**，reach_bottle/lift 指标均为0。运行链路通过不等于任务成功，也不证明新解码器已提升正式测评成绩。
- requested_seed=42，但 official 初始化实际使用 seed=0、`episode_55.hdf5`、randomize_idx=7610；结果文件已记录，未手动改写官方初始化。
- 正式MP4为1280×720、400帧、4,266,067字节、26.667秒；ffprobe 通过，完整 ffmpeg 解码退出0。
- `episodes.jsonl` 确有 completed episode，`result.json`、`metrics.json` 均存在；日志未发现 WsError、Traceback、paired12几何门失败或所检查的hold错误标记。上游MANO已有的显式fallback仍有触发，未更改其门槛。
- 运行结束后测试policy server进程已退出。没有启动93回合，没有切换默认部署。

正式证据目录：`artifacts/egovla_paired12_20260908/official_single/`；命令脚本 `official_single.sh`，启动时由用户授权的 `ACCEPT_EULA=Y` 环境变量传入。

|正式文件|SHA256|
|---|---|
|result.json|98347a377878bb056c459336c93e0547250e2cd43f80e401c3fa12bfb5e8549b|
|episodes.jsonl|fc109427b6b6e1a0a4cc35ed287ea441fe866914d2293eb46293a2e214e204ed|
|metrics.json|e9f5e7296d2699e935745605678846ce3b33063d535a87e59da309aafa6f41ba|
|videos/episode_000000_cam_head.mp4|95bdb8b42a0939156880438fe29d04cc2cae405223cc73c49ea7c1d246d8d2dd|

证据：远端 artifact 目录下 `debug_dry_run.json`、`debug_server.log`、`debug_client.log`、`debug_rollout.sh` 和 `debug_rollout/`。许可阻止的旧日志以 `.eula_blocked.log` 保留。解码器保持显式可选，未切换默认部署。

## 重训练方案

若需要真实12D无损语义：数据转换应同时保存全部12维观测角与任务定义的12维目标动作，明确区分目标控制量与下一帧实际状态；不再用 Spark mimic 替代独立关节。保留现有 EE 腕部/视觉骨干，可新增真实 H1 12D 手部头并微调，重新计算其统计量、维度metadata和checkpoint绑定。该新表示不能伪装为旧61/51配置兼容，也不能靠旧checkpoint逆解码保证恢复已丢失信息。
