import { useCallback, useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  Brain,
  Filter,
  Link2,
  ListFilter,
  Radio,
  Timer,
  Zap,
} from 'lucide-react';
import { agentAPI, type AgentEvent, type LLMTelemetry, type WorkflowSummary } from '../lib/api';
import { cn, formatDateTime } from '../lib/utils';
import { useWebSocket, type WebSocketMessage } from '../hooks/useWebSocket';
import { buildWebSocketUrl } from '../lib/websocket';

const EVENT_TYPE_LABELS: Record<string, string> = {
  intent: 'Intent',
  plan_generated: 'Plan Generated',
  plan_judged: 'Plan Judged',
  trigger_fired: 'Trigger Fired',
  order_submitted: 'Order Submitted',
  fill: 'Fill',
  trade_blocked: 'Trade Blocked',
  position_update: 'Position Update',
  tick: 'Market Tick',
};

const SOURCE_OPTIONS = ['broker_agent', 'execution_agent', 'judge_agent', 'execution_ledger', 'market_stream', 'ops_api'];

function formatTokens(inCount?: number, outCount?: number) {
  const inVal = Number.isFinite(inCount) ? (inCount as number) : 0;
  const outVal = Number.isFinite(outCount) ? (outCount as number) : 0;
  return `${inVal.toLocaleString()} in / ${outVal.toLocaleString()} out`;
}

function formatCost(cost?: number) {
  const val = Number.isFinite(cost) ? (cost as number) : 0;
  return `$${val.toFixed(4)}`;
}

function renderEventSummary(event: AgentEvent) {
  const { type } = event;
  const payload = event.payload || {};
  switch (type) {
    case 'intent':
      return payload.text || 'User intent';
    case 'plan_generated':
      return `Plan for ${payload.symbol || 'unknown'} (${payload.strategy_id || 'n/a'})`;
    case 'plan_judged':
      return `Score ${payload.overall_score ?? 'n/a'} / 100`;
    case 'order_submitted':
      return `${payload.side || ''} ${payload.qty ?? '?'} ${payload.symbol || ''} @ ${
        payload.type || 'market'
      }`;
    case 'fill':
      return `${payload.side || ''} ${payload.qty ?? '?'} ${payload.symbol || ''} @ ${
        payload.price ?? '?'
      }`;
    case 'trade_blocked':
      return payload.reason || 'Blocked';
    case 'position_update':
      return `${payload.symbol || ''} qty ${payload.qty ?? '?'} pnl ${payload.pnl ?? 'n/a'}`;
    case 'trigger_fired':
      return `${payload.symbol || ''} ${payload.side || ''} trigger ${payload.trigger_id || ''}`;
    default:
      return JSON.stringify(payload);
  }
}

export function AgentInspector() {
  const [typeFilter, setTypeFilter] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [runIdFilter, setRunIdFilter] = useState('');
  const [correlationFilter, setCorrelationFilter] = useState('');
  const [limit, setLimit] = useState(50);
  const [liveEvents, setLiveEvents] = useState<AgentEvent[]>([]);

  const {
    data: events = [],
    isLoading: eventsLoading,
    isFetching: eventsFetching,
  } = useQuery({
    queryKey: [
      'agent-inspector-events',
      typeFilter,
      sourceFilter,
      runIdFilter,
      correlationFilter,
      limit,
    ],
    queryFn: () =>
      correlationFilter
        ? agentAPI.getEventChain(correlationFilter)
        : agentAPI.getEvents({
            type: typeFilter || undefined,
            source: sourceFilter || undefined,
            run_id: runIdFilter || undefined,
            correlation_id: correlationFilter || undefined,
            limit,
          }),
    refetchInterval: 4000,
  });

  const { data: telemetry = [], isLoading: telemetryLoading } = useQuery({
    queryKey: ['agent-inspector-llm-telemetry'],
    queryFn: () => agentAPI.getLLMTelemetry(),
    refetchInterval: 8000,
  });

  const { data: workflows = [], isLoading: workflowsLoading } = useQuery({
    queryKey: ['agent-inspector-workflows'],
    queryFn: () => agentAPI.listWorkflows(),
    refetchInterval: 8000,
  });

  const workflowList = Array.isArray(workflows) ? workflows : [];

  // Reset live buffer when filters change to avoid showing mismatched items
  useEffect(() => {
    setLiveEvents([]);
  }, [typeFilter, sourceFilter, runIdFilter, correlationFilter]);

  const matchesFilters = useCallback(
    (evt: AgentEvent) => {
      if (typeFilter && evt.type !== typeFilter) return false;
      if (sourceFilter && evt.source !== sourceFilter) return false;
      if (runIdFilter && evt.run_id !== runIdFilter) return false;
      if (correlationFilter && evt.correlation_id !== correlationFilter) return false;
      return true;
    },
    [typeFilter, sourceFilter, runIdFilter, correlationFilter]
  );

  const handleWebSocketMessage = useCallback(
    (message: WebSocketMessage) => {
      const evt: AgentEvent = {
        event_id: message.event_id,
        timestamp: message.timestamp,
        source: message.source,
        type: message.type,
        payload: message.payload,
        run_id: message.run_id,
        correlation_id: message.correlation_id,
      };

      if (!matchesFilters(evt)) {
        return;
      }

      setLiveEvents((prev) => {
        const next = [evt, ...prev];
        const seen = new Set<string>();
        const deduped: AgentEvent[] = [];
        for (const e of next) {
          if (seen.has(e.event_id)) continue;
          seen.add(e.event_id);
          deduped.push(e);
          if (deduped.length >= 200) break;
        }
        return deduped;
      });
    },
    [matchesFilters]
  );

  const { isConnected: wsConnected } = useWebSocket(
    buildWebSocketUrl('/ws/live'),
    {},
    handleWebSocketMessage
  );

  const displayEvents = useMemo(() => {
    const merged = [...liveEvents, ...events];
    const seen = new Set<string>();
    const deduped: AgentEvent[] = [];
    for (const evt of merged) {
      if (seen.has(evt.event_id)) continue;
      seen.add(evt.event_id);
      deduped.push(evt);
    }
    return deduped;
  }, [liveEvents, events]);

  const sortedTelemetry = useMemo<LLMTelemetry[]>(
    () =>
      [...telemetry].sort(
        (a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime()
      ),
    [telemetry]
  );

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <Brain className="w-6 h-6 text-blue-500" />
            Agent Inspector
          </h1>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Trace decision chains, monitor LLM costs, and check workflow health in one place.
          </p>
        </div>
        {eventsFetching && (
          <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <Activity className="w-4 h-4 animate-spin" />
            Refreshing…
          </div>
        )}
        <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
          <span
            className={cn(
              'inline-flex items-center gap-1 px-2 py-1 rounded-full',
              wsConnected
                ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-200'
                : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-200'
            )}
          >
            <Radio className="w-3 h-3" />
            {wsConnected ? 'Live feed' : 'Live feed (polling fallback)'}
          </span>
        </div>
      </div>

      {/* Event Stream */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-100 dark:border-gray-700">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <ListFilter className="w-4 h-4 text-blue-500" />
              Decision Chain
              {correlationFilter && (
                <span className="ml-2 text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-200">
                  Correlation: {correlationFilter}
                </span>
              )}
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Filter by type/source/run and trace correlation IDs end-to-end.
            </p>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap gap-2">
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Filter className="w-4 h-4" />
              Filters
            </div>
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="">All Types</option>
              {Object.entries(EVENT_TYPE_LABELS).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="">All Sources</option>
              {SOURCE_OPTIONS.map((src) => (
                <option key={src} value={src}>
                  {src}
                </option>
              ))}
            </select>
            <input
              value={runIdFilter}
              onChange={(e) => setRunIdFilter(e.target.value)}
              placeholder="Run ID"
              className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <input
              value={correlationFilter}
              onChange={(e) => setCorrelationFilter(e.target.value)}
              placeholder="Correlation"
              className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              {[25, 50, 100].map((v) => (
                <option key={v} value={v}>
                  {v} events
                </option>
              ))}
            </select>
            {correlationFilter && (
              <button
                className="text-xs text-blue-600 dark:text-blue-300 underline"
                onClick={() => setCorrelationFilter('')}
              >
                Clear correlation
              </button>
            )}
          </div>
        </div>

        <div className="divide-y divide-gray-200 dark:divide-gray-700 max-h-[520px] overflow-y-auto">
          {eventsLoading && displayEvents.length === 0 ? (
            <div className="p-6 text-sm text-gray-500">Loading events…</div>
          ) : displayEvents.length === 0 ? (
            <div className="p-6 text-sm text-gray-500">No events found.</div>
          ) : (
            displayEvents.map((event) => (
              <div
                key={event.event_id}
                className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/40 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <div className="flex flex-col items-center gap-2">
                    <div className="p-2 rounded-full bg-blue-50 dark:bg-blue-900/40 text-blue-600 dark:text-blue-300">
                      <Radio className="w-4 h-4" />
                    </div>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                            {EVENT_TYPE_LABELS[event.type] || event.type}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {event.source}
                          </span>
                          {event.correlation_id && (
                            <span className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200">
                              Corr: {event.correlation_id}
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {formatDateTime(event.timestamp)}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {event.correlation_id && (
                          <button
                            onClick={() => setCorrelationFilter(event.correlation_id || '')}
                            className="text-xs inline-flex items-center gap-1 px-2 py-1 rounded bg-blue-50 text-blue-700 dark:bg-blue-900/40 dark:text-blue-200 hover:bg-blue-100 dark:hover:bg-blue-900/60"
                          >
                            <Link2 className="w-3 h-3" />
                            Trace chain
                          </button>
                        )}
                      </div>
                    </div>

                    <div className="mt-2 text-sm text-gray-800 dark:text-gray-200">
                      {renderEventSummary(event)}
                    </div>

                    <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-2 text-xs text-gray-600 dark:text-gray-400">
                      {event.run_id && (
                        <div className="flex items-center gap-1">
                          <Zap className="w-3 h-3 text-yellow-500" />
                          Run: {event.run_id}
                        </div>
                      )}
                      <div className="md:col-span-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-700 rounded p-2 overflow-x-auto">
                        <pre className="whitespace-pre-wrap break-words text-[11px] text-gray-900 dark:text-gray-100">
                          {JSON.stringify(event.payload, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* LLM Telemetry + Workflow Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-100 dark:border-gray-700">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-500" />
                LLM Telemetry
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Model usage, token counts, and cost estimates.
              </p>
            </div>
            {telemetryLoading && (
              <Timer className="w-4 h-4 text-gray-400 animate-spin" />
            )}
          </div>

          <div className="max-h-[420px] overflow-y-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-900 text-gray-600 dark:text-gray-300">
                <tr>
                  <th className="px-4 py-2 text-left font-semibold">Run</th>
                  <th className="px-4 py-2 text-left font-semibold">Model</th>
                  <th className="px-4 py-2 text-left font-semibold">Tokens</th>
                  <th className="px-4 py-2 text-left font-semibold">Cost</th>
                  <th className="px-4 py-2 text-left font-semibold">Duration</th>
                  <th className="px-4 py-2 text-left font-semibold">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {sortedTelemetry.length === 0 ? (
                  <tr>
                    <td
                      colSpan={6}
                      className="px-4 py-6 text-center text-gray-500 dark:text-gray-400"
                    >
                      No telemetry yet.
                    </td>
                  </tr>
                ) : (
                  sortedTelemetry.map((entry, idx) => (
                    <tr
                      key={`${entry.ts}-${idx}`}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700/40"
                    >
                      <td className="px-4 py-2 text-gray-900 dark:text-gray-100">
                        {entry.run_id || 'n/a'}
                      </td>
                      <td className="px-4 py-2 text-gray-900 dark:text-gray-100">
                        {entry.model}
                      </td>
                      <td className="px-4 py-2 text-gray-900 dark:text-gray-100">
                        {formatTokens(entry.tokens_in, entry.tokens_out)}
                      </td>
                      <td className="px-4 py-2 text-gray-900 dark:text-gray-100">
                        {formatCost(entry.cost_estimate)}
                      </td>
                      <td className="px-4 py-2 text-gray-900 dark:text-gray-100">
                        {Number.isFinite(entry.duration_ms) ? `${entry.duration_ms} ms` : 'n/a'}
                      </td>
                      <td className="px-4 py-2 text-gray-900 dark:text-gray-100">
                        {entry.ts ? formatDateTime(entry.ts) : 'n/a'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-100 dark:border-gray-700">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center gap-2">
            <Activity className="w-4 h-4 text-green-500" />
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Workflow Status
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Broker, Execution, and Judge state.
              </p>
            </div>
          </div>

          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {workflowsLoading && workflowList.length === 0 ? (
              <div className="p-4 text-sm text-gray-500">Loading workflows…</div>
            ) : workflowList.length === 0 ? (
              <div className="p-4 text-sm text-gray-500">No workflows found.</div>
            ) : (
              workflowList.map((wf: WorkflowSummary) => (
                <div key={wf.run_id} className="p-4 flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {wf.run_id}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Updated {formatDateTime(wf.last_updated)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Mode: {wf.mode}
                    </div>
                  </div>
                  <span
                    className={cn(
                      'px-2 py-1 rounded text-xs font-semibold capitalize',
                      wf.status === 'running' && 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-200',
                      wf.status === 'paused' && 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-200',
                      wf.status === 'stopped' && 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200'
                    )}
                  >
                    {wf.status}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
