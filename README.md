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
股票池。支持 `sample`、当前沪深300快照 `csi300` 和当前标普500快照
`sp500`。

## 安装与运行

需要本地 MongoDB 服务。建议在已有 `quant` Conda 环境中安装：

```bash
conda run -n quant pip install -e ".[test]"
```

也可以根据仓库保存的环境文件创建完整开发与 Notebook 环境：

```bash
conda env create -f environment.yml
conda activate quant
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

同步当前沪深300和标普500成分股：

```bash
conda run -n quant quant-backtest sync-pools --pool all
```

当前成分股只从同步日开始生效，并非历史成分。使用它们研究同步日前的历史会产生
存活者偏差，因此 CLI 会拒绝把快照成员伪装成更早日期的历史成员。

下载 2015 年至上个月末的数据；再次运行时从各股票最新日期后增量更新：

```bash
conda run -n quant quant-backtest load-data
```

指数股票池默认不自动下载全部行情，可按批次执行：

```bash
conda run -n quant quant-backtest load-data \
  --pool csi300 --batch-offset 0 --batch-size 25 --request-delay 0.5
```

每只股票的下载错误会单独记录，不会中断整个批次。

运行等权复合因子、月初等权调仓示例：

```bash
conda run -n quant quant-backtest backtest \
  --start 2020-01-01 --end 2024-12-31 \
  --factor composite --lookback 20 \
  --reversal-lookback 5 --volatility-window 20 --top-n 5 \
  --initial-cash 1000000 --max-volume-pct 10
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

## 成交模型

CLI 默认启用：

- A 股双边佣金 3 bps，卖出印花税 5 bps。
- 美股双边佣金 1 bp，无卖出印花税。
- 双边固定滑点 5 bps。
- 单日成交额不超过当日成交额的 10%。
- 缺价、成交量缺失或成交量为 0 时不成交，原持仓保留。
- 买入受可用现金约束，卖出不超过已有持仓。

调仓目标在月初首个交易日生成，下一交易日尝试一次成交，未成交部分不跨日追单。
结果包含实际持仓、交易明细、换手率、佣金、税费、滑点和现金。

如需复现原来的无摩擦权重回测：

```bash
conda run -n quant quant-backtest backtest \
  --pool sample --start 2020-01-01 --end 2024-12-31 --frictionless
```

AkShare 的 A 股成交量原始单位为“手”，框架入库时转换为“股”。旧版已经入库的
A 股数据需要执行 `load-data --pool sample --full` 全量刷新后，成交量约束才有
统一口径。

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

## 价格图表与 Notebook

安装 Notebook 环境：

```bash
conda run -n quant pip install -e ".[test,notebook]"
conda run -n quant jupyter lab notebooks/price_chart_demo.ipynb
```

Notebook 的参数单元可选择 `CODE`、起止日期、`line/candlestick`、价格字段、
是否复权、K 线频率、成交量副图和 PNG 输出路径。折线图默认使用 `close`。

也可以直接调用绘图引擎：

```python
from quant_backtest import PriceChartEngine

engine = PriceChartEngine()
figure, axes = engine.plot(
    "AAPL.US",
    "2023-01-01",
    "2024-12-31",
    chart_type="candlestick",
    frequency="auto",
    show_volume=True,
)
```

K 线自动聚合规则：

- 不超过 160 根：日 K。
- 161–750 根：周 K。
- 超过 750 根：月 K。

标题会显示实际频率。折线图保留全部行情点，并自动调整画布和日期刻度。设置
`adjusted=True` 时，OHLC 会乘以 `adj_factor`。

## 双均线策略回测 Notebook

```bash
conda run -n quant jupyter lab notebooks/moving_average_backtest_demo.ipynb
```

Notebook 会优先读取 MongoDB；如果所选日期范围内缺少策略标的或基准行情，会
通过 AkShare 或 yfinance 自动下载、清洗并写入数据库，无需预先运行数据命令。
A 股可以使用 `510300.SH` 作为沪深300 ETF 基准。参数单元可配置标的、基准、
日期、短长均线窗口、单边交易成本、无风险利率和图片输出路径。

双均线在短均线高于长均线时持有标的，否则保持空仓。收盘信号从下一交易日生效，
避免同日未来数据；回测输出策略与基准的累计收益、年化收益、夏普比率、最大回撤、
跟踪误差和信息比率，并绘制交易点、净值/超额收益及回撤对比。

首版是单标的多头/空仓研究模型，交易成本按仓位变化以基点计，不模拟做空、杠杆、
最低佣金、成交量限制或订单簿。

## 测试

```bash
conda run -n quant pytest
```

测试使用固定数据和 mongomock，不访问实时行情网络，也不会修改本地
`quant_db`。覆盖数据清洗、幂等 upsert、股票池读取、增量日期、未来数据
隔离、指数成分标准化、三类因子、截面处理、复合因子、费用、滑点、停牌、
成交量限制、月度调仓、双均线策略、价格绘图、Notebook 参数和绩效指标。当前
预期结果为 `57 passed`。

## 首版限制

- yfinance 和 AkShare 是免费数据源，接口稳定性与数据授权需由使用者确认。
- 沪深300和标普500目前只有当前快照；无偏历史研究仍需导入历史成分。
- 首版不模拟最低佣金、A 股整手、涨跌停和订单跨日排队。
- 增量加载不会主动重算已入库历史复权因子；发生公司行动后可用
  `load-data --full` 全量刷新。
