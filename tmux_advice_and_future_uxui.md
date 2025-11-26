Totally fair—the current tmux layout dumps a firehose of mixed logs, so tracking holdings is painful. A few ideas to
  make it livable right away, plus some medium-term improvements:

  Quick tmux Wins

  - Ctrl-b z zooms the focused pane; use it on broker or ticker panes when you need to read through output, then toggle
    it off.
  - Ctrl-b [ enters copy/scroll mode, letting you navigate a single pane’s backlog without the others racing by. Combine
    with / search to jump to “Portfolio” lines.
  - Move panes you interact with frequently (broker, ticker_ui) to the top with tmux swap-pane; or edit run_stack.sh so
    they launch in the first row.
  - Redirect noisy panes to files instead: wrap commands like python agents/execution_agent_client.py | tee logs/
    execution.log so you can tail them separately.

  Better Holdings Visibility (Short Term)

  - Broker already prints snapshots when prompted; make it part of your workflow to type portfolio between runs to get a
    fresh view.
  - The ticker_ui_service.py pane can be zoomed for a cleaner stream of balances; consider tweaking it to refresh less
    frequently or collapse verbose sections.
  - Add a “status bar” tmux pane at the bottom via tmux split-window -b running watch -n5 uv run python tools/
    portfolio_summary.py (you’d need to build that script once, aggregating holdings). Keeps the focus on key metrics.

  Medium-Term Enhancements

  - Create a dedicated CLI command (e.g., uv run python agents/show_holdings.py) that fetches and prints portfolio state
    in a compact table. Trigger it from tmux on demand or bind it to a key.
  - Introduce structured logging: configure logging_utils.setup_logging to output JSON or key-value pairs, then run logan/
    jq filters to show only balance updates or errors.
  - Build a lightweight textual dashboard (Textual/Urwid/Blessed) that subscribes to the shared bus and renders holdings,
    PnL, exposure, and alerts in one view.
  - If you lean into Docker, expose the Temporal UI (http://localhost:8088) and integrate a small FastAPI endpoint that
    broadcasts portfolio snapshots; then any browser-based dashboard can poll or subscribe without diving into tmux.
  - Longer term, consider aggregating agent chatter in a log viewer (Loki/Grafana or even stern|k9s style) so raw stdout
    doesn’t dominate the interface.

  If you want to pursue one of these (like adding a summary command or restructuring the tmux layout), happy to help
  sketch or implement it.[]