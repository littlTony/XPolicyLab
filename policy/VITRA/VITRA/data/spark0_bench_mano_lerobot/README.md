# Spark-0 MANO LeRobot v3.0

This dataset augments the original 600-episode Spark-0 LeRobot export with the fitted MANO arrays from `/mnt/xspark-data/zijian/Spark_0/data/spark0_bench_mano`.

- Rows: **150,810** (`T-1` per episode)
- Episodes: **600**
- FPS: **25**
- Robot label: **tianji_marvin_wujihand2_mano**.
- Existing Wuji fields and camera videos are unchanged.
- MANO leaves are separate `float64` columns under `observation.mano.*` and `action.mano.*`.
- The source MANO action is already `state[t+1]`; it was not shifted again.
- The valid final observation of each episode is stored in `meta/mano_terminal_states.parquet`.
- See `meta/mano.json` before converting the 45-D rotation vectors to VITRA Euler-xyz.

The MANO xyz values are verified to equal the source Wuji **Link7 EE** xyz; they are not an independently calibrated physical-wrist translation. `valid` means finite and source validity-mask true, not that the geometric retarget residual passed a threshold.
