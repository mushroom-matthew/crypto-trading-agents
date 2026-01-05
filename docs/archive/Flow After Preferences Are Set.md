Flow After Preferences Are Set

  - The broker LLM asks for the three core fields, packages them into {"preferences": {...}}, and invokes the
    set_user_preferences MCP tool; the tool_result_data helper echoes the server’s confirmation back into the broker
    conversation so the assistant can acknowledge success (agents/broker_agent_client.py:142,183; agents/utils.py:95).
  - set_user_preferences validates the payload, ensures the broker workflow exists, and then signals four Temporal
    workflows—broker, execution, judge, and ledger—so every component has the fresh snapshot (mcp_server/app.py:158-237).
  - BrokerAgentWorkflow persists those values for future queries and defaults new sessions to the updated mix, making
    get_user_preferences a single point of truth for the rest of the stack (agents/workflows/broker_agent_workflow.py:13-
    38).

  Component Reactions

  - Execution workflow caches the dict and exposes it via get_user_preferences; a background task in the execution client
    polls that query every 2 s and logs any change, keeping the in-memory copy current even if the workflow restarts
    (agents/workflows/execution_agent_workflow.py:29-40; agents/execution_agent_client.py:129-147).
  - The execution client also watches for judge-driven prompt changes every 5 s; once the workflow’s system_prompt
    changes, the first conversation message is replaced so the next LLM call runs with the updated instructions (agents/
    execution_agent_client.py:150-171).
  - Judge workflow stores the preferences and the judge client polls them every 3 s; on change it immediately calls
    update_prompt_for_user_preferences, which pulls the current execution prompt, has the LLM rewrite it to reflect
    the new risk/style baseline, signals the execution workflow, and logs the preference_update action through
    AgentLogger (agents/workflows/judge_agent_workflow.py:43-48; agents/judge_agent_client.py:775-809,497-558; tools/
    agent_logger.py:101-150).
  - The ledger workflow parses optional keys like profit_scraping_percentage (string or numeric) and updates the scraping
    ratio used on profitable sells, ensuring cash segregation honours the user’s preference going forward (agents/
    workflows/execution_ledger_workflow.py:35-108).

  Downstream Behaviour

  - Every nudge cycle, the execution agent collects deterministic context in parallel—ticks, portfolio, performance, and
    the latest preferences via the MCP get_user_preferences tool—before handing the bundle to the LLM, so order sizing and
    commentary always reflect the stored settings (agents/execution_agent_client.py:247-310,329-366).
  - Logged decisions embed the preference snapshot and get pushed to the execution workflow (and mirrored to JSONL) so
    audits show which profile drove each trade (tools/agent_logger.py:101-150).
  - Judge evaluations call get_user_preferences again to build baselines for scoring and to decide whether prompt/
    context adjustments are needed, keeping the performance feedback loop aligned with the user profile (agents/
    judge_agent_client.py:320-404).

  Edge Cases & Guarantees

  - If the execution or judge workflows aren’t running yet, the initial signals are best-effort (warnings are logged), but
    the broker workflow retains the preferences so subsequent get_user_preferences queries still return the right values
    once agents come online (mcp_server/app.py:213-231).
  - All of these updates are durable in Temporal; after any restart the workflows replay their last state, the polling
    tasks detect the stored preferences, and the LLM prompts are re-synchronized automatically.

Trade Lifecycle

  - Temporal nudges fire about every 25 s, kicking off the execution agent loop (agents/execution_agent_client.py:220-
    290). Before any LLM call, the agent fetches fresh ticks, portfolio status, saved preferences, performance
    metrics, and risk metrics in parallel so the context handed to the model reflects the latest state (agents/
    execution_agent_client.py:300-362).
  - That bundle is appended to the conversation and the Langfuse-instrumented OpenAI client decides whether
    to call place_mock_order; only that tool is exposed, keeping strategy selection deterministic (agents/
    execution_agent_client.py:369-425).
  - When the LLM requests orders, the MCP server validates cash and positions, executes the Temporal workflow
    PlaceMockBatchOrder, and forwards the fills to the ledger (mcp_server/app.py:304-427). The ledger records timestamped
    transactions, updates positions, and applies profit-scraping logic for profitable sells (agents/workflows/
    execution_ledger_workflow.py:34-116).
  - Every decision is logged with the full input context, including the user’s preferences, so the workflow history
    (and the JSONL debug files) show exactly why each trade happened (agents/execution_agent_client.py:488-526, tools/
    agent_logger.py:101-150).

  Performance Evaluation

  - The judge agent waits for either a scheduled window or an immediate trigger, then queries the ledger for performance
    metrics, risk metrics, and recent transactions (agents/judge_agent_client.py:668-703). It synthesizes those via
    PerformanceAnalyzer and a second LLM pass for decision-quality scoring (agents/judge_agent_client.py:194-314, tools/
    performance_analysis.py:1-320).
  - Scores and recommendations feed determine_context_updates; if user preferences changed or performance slipped below
    the threshold, the judge rewrites the execution prompt with the user’s risk/style baked in, then signals the execution
    workflow to swap prompts (agents/judge_agent_client.py:355-560, agents/workflows/execution_agent_workflow.py:42-46).
  - Completed evaluations are persisted in the judge workflow for trend queries and logged via AgentLogger, giving you
    both a Temporal audit trail and file-based records of every review (agents/workflows/judge_agent_workflow.py:17-111,
    agents/judge_agent_client.py:724-752).

  Portfolio & Market Monitoring

  - Portfolio cash, positions, entry prices, and PnL are always sourced from the ledger workflow through the MCP
    get_portfolio_status call, so any agent or UI asking for holdings sees a single, authoritative snapshot (mcp_server/
    app.py:488-519).
  - The same execution cycle pulls get_risk_metrics and get_performance_metrics, letting the LLM weigh concentration, cash
    ratios, drawdown, etc., before placing trades (agents/execution_agent_client.py:310-362).
  - Market streaming is symbol-driven: the broker signals set_symbols, start_market_stream spins up SubscribeCEXStream,
    and downstream feature workflows keep appending ticks for those pairs (mcp_server/app.py:122-156, tools/
    market_data.py:97-205). If you add a position in a new pair, that symbol must be in the active set; otherwise the
    ledger will warn about stale prices and future monitoring enhancements should auto-enroll newly held symbols (agents/
    workflows/execution_ledger_workflow.py:117-156).
  - Risk-logging helpers flag price staleness and maintain per-symbol timestamps, so even in multi-market
    scenarios the system can detect if a stream dropped and surface the gap during evaluations (agents/workflows/
    execution_ledger_workflow.py:117-156, agents/judge_agent_client.py:682-722).

  If you want better visibility across many markets, the next step would be to auto-sync the symbol list with the ledger’s
  open positions or add a watch service that subscribes whenever a new holding appears.

   Nothing deterministic executes trades today—the execution agent hands the full market/portfolio context to the Langfuse-
  instrumented OpenAI client every nudge and lets the model decide whether to call place_mock_order. There is no hand-
  coded signal, indicator, or rule beyond what the prompt describes.

  Here’s what exists in code:

  - tools/strategy_signal.py:6-31 only logs an incoming “momentum” signal; it doesn’t compute any indicator or drive the
    execution loop.
  - tools/feature_engineering.py:101-287 maintains rolling tick history per symbol (ComputeFeatureVector) and exposes
    helpers like latest_price, but the vector/indicator values are not consumed anywhere—EvaluateStrategyMomentum never
    uses them.
  - agents/execution_agent_client.py:300-530 is the core decision loop. After collecting ticks, get_performance_metrics,
    get_risk_metrics, etc., it appends that JSON to the conversation and waits for the LLM to issue orders. There is no
    deterministic scoring or threshold; everything hinges on the prompt text and preferences.
  - Performance metrics do exist: the ledger computes totals, PnL, drawdown, cash ratios, and basic averages (agents/
    workflows/execution_ledger_workflow.py:200-347), and tools/performance_analysis.py:1-320 derives Sharpe, win rate,
    decision-quality scores, etc. Those metrics feed the judge for evaluations and prompt tuning, but they aren’t tied to
    any rule-based trading logic.

  So the current “strategy” is effectively the LLM interpreting the system prompt plus user preferences to decide what
  to do. If you want concrete signals (e.g., moving-average crossovers, RSI bands, multi-symbol filters), you’d need to
  implement them—likely in the feature workflows or a new tools/strategies/* module—and surface the results either as
  deterministic orders or as structured inputs the LLM must honor.
