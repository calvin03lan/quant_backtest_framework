# 简易量化回测框架

Python 实现的最小因子回测框架，数据存储使用 MongoDB。A 股行情来自
AkShare，美股行情来自 yfinance。

## 数据结构

`quant_db.daily_ohlc` 保存：

`date, code, market, open, high, low, close, adj_factor, volume`

- `date` 为 MongoDB 日期类型。
- `(code, date)` 是唯一键，重复更新采用 upsert。
- OHLC 保存原始价格；复权价格为 `close * adj_factor`。
- 示例股票池为 5 只 A 股和 5 只美股，位于
  `quant_backtest.cli.SAMPLE_CODES`。

`quant_db.stock_pool` 使用 `pool_id/code/valid_from/valid_to` 保存带生效区间的
股票池。

## 安装与运行

需要本地 MongoDB 服务。建议在已有 `quant` Conda 环境中安装：

```bash
conda run -n quant pip install -e ".[test]"
```

可通过环境变量覆盖连接配置：

```bash
export MONGO_URI=mongodb://localhost:27017
export MONGO_DB=quant_db
```

初始化索引和示例股票池：

```bash
conda run -n quant quant-backtest init-db
```

下载 2015 年至上个月末的数据；再次运行时从各股票最新日期后增量更新：

```bash
conda run -n quant quant-backtest load-data
```

运行等权复合因子、月初等权调仓示例：

```bash
conda run -n quant quant-backtest backtest \
  --start 2020-01-01 --end 2024-12-31 \
  --factor composite --lookback 20 \
  --reversal-lookback 5 --volatility-window 20 --top-n 5
```

只运行原有动量因子：

```bash
conda run -n quant quant-backtest backtest \
  --start 2020-01-01 --end 2024-12-31 \
  --factor momentum --lookback 20 --top-n 5
```

预期输出为 `net_value`、`annual_return`、`max_drawdown` 和
`sharpe_ratio`。调仓目标在月初首个交易日收盘形成，从下一交易日起生效，
避免使用尚不可交易的当日收盘信息。

## 因子层

- 动量因子：过去 N 条有效行情记录的复权收益，值越高越优。
- 短期反转因子：过去 N 条有效行情记录的复权收益取反。
- 低波动因子：复权日收益的 N 条记录滚动波动率取反。
- 所有原始因子都滞后一条行情记录，因此日期 T 的信号最多使用 T-1
  及更早的数据。
- 复合因子按交易日分别进行 3 倍 MAD 去极值和 z-score 标准化，再按配置
  权重合成；默认三类因子等权。
- 窗口 N 按每只股票的有效行情行数计算，不是日历天数。存在重复
  `(code, date)` 的输入会被拒绝。

## 测试

```bash
conda run -n quant pytest
```

测试使用固定数据和 mongomock，不访问实时行情网络，也不会修改本地
`quant_db`。覆盖数据清洗、幂等 upsert、股票池读取、增量日期、未来数据
隔离、三类因子、截面处理、复合因子、月度调仓和绩效指标。当前预期结果为
`24 passed`。

## 首版限制

- yfinance 和 AkShare 是免费数据源，接口稳定性与数据授权需由使用者确认。
- 股票池示例不是历史指数成分，正式研究应导入带生效区间的历史成分股。
- 首版不模拟手续费、滑点、成交量约束和复杂订单撮合。
- 增量加载不会主动重算已入库历史复权因子；发生公司行动后可用
  `load-data --full` 全量刷新。
