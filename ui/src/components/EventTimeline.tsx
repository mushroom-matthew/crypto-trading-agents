import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import {
  MessageSquare,
  Lightbulb,
  Gavel,
  Send,
  CheckCircle,
  XCircle,
  Filter,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from 'lucide-react';
import { agentAPI, type AgentEvent } from '../lib/api';
import { cn, formatDateTime } from '../lib/utils';

// Event type configurations
const EVENT_CONFIG: Record<
  string,
  { icon: React.ComponentType<any>; color: string; label: string }
> = {
  intent: {
    icon: MessageSquare,
    color: 'text-blue-500',
    label: 'User Intent',
  },
  plan_generated: {
    icon: Lightbulb,
    color: 'text-yellow-500',
    label: 'Plan Generated',
  },
  plan_judged: {
    icon: Gavel,
    color: 'text-purple-500',
    label: 'Plan Judged',
  },
  order_submitted: {
    icon: Send,
    color: 'text-indigo-500',
    label: 'Order Submitted',
  },
  fill: {
    icon: CheckCircle,
    color: 'text-green-500',
    label: 'Trade Filled',
  },
  trade_blocked: {
    icon: XCircle,
    color: 'text-red-500',
    label: 'Trade Blocked',
  },
  llm_call: {
    icon: Sparkles,
    color: 'text-amber-500',
    label: 'LLM Call',
  },
};

interface EventTimelineProps {
  limit?: number;
  showFilter?: boolean;
  runId?: string;
}

export function EventTimeline({ limit = 50, showFilter = true, runId }: EventTimelineProps) {
  const [filterType, setFilterType] = useState<string>('');
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());

  // Query agent events with auto-refresh every 3 seconds
  const { data: events = [], isLoading } = useQuery({
    queryKey: ['agent-events', filterType, runId],
    queryFn: () =>
      agentAPI.getEvents({
        type: filterType || undefined,
        run_id: runId || undefined,
        limit,
      }),
    refetchInterval: 3000,
  });

  const toggleExpanded = (eventId: string) => {
    setExpandedEvents((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(eventId)) {
        newSet.delete(eventId);
      } else {
        newSet.add(eventId);
      }
      return newSet;
    });
  };

  const eventTypes = Object.keys(EVENT_CONFIG);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Event Timeline
          </h2>

          {/* Filter */}
          {showFilter && (
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-gray-500" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="text-sm border border-gray-300 dark:border-gray-600 rounded px-3 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="">All Events</option>
                {eventTypes.map((type) => (
                  <option key={type} value={type}>
                    {EVENT_CONFIG[type].label}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Event List */}
      <div className="max-h-96 overflow-y-auto">
        {isLoading && events.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <p>Loading events...</p>
          </div>
        ) : events.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <p>No events found</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {events.map((event) => {
              const config = EVENT_CONFIG[event.type] || {
                icon: MessageSquare,
                color: 'text-gray-500',
                label: event.type,
              };
              const Icon = config.icon;
              const isExpanded = expandedEvents.has(event.event_id);

              return (
                <div
                  key={event.event_id}
                  className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <div className="flex items-start gap-3">
                    {/* Icon */}
                    <div
                      className={cn(
                        'flex-shrink-0 p-2 rounded-lg bg-gray-100 dark:bg-gray-700',
                        config.color
                      )}
                    >
                      <Icon className="w-4 h-4" />
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      {/* Header */}
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">
                              {config.label}
                            </span>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {event.source}
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                            {formatDateTime(event.timestamp)}
                          </div>
                        </div>

                        {/* Expand/Collapse button */}
                        <button
                          onClick={() => toggleExpanded(event.event_id)}
                          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
                        >
                          {isExpanded ? (
                            <ChevronUp className="w-4 h-4 text-gray-500" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-gray-500" />
                          )}
                        </button>
                      </div>

                      {/* Summary (always visible) */}
                      <div className="mt-2 text-sm text-gray-700 dark:text-gray-300">
                        {renderEventSummary(event)}
                      </div>

                      {/* Expanded details */}
                      {isExpanded && (
                        <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-900 rounded text-xs font-mono">
                          <div className="space-y-1">
                            {event.run_id && (
                              <div>
                                <span className="text-gray-500">Run ID:</span>{' '}
                                <span className="text-gray-900 dark:text-gray-100">
                                  {event.run_id}
                                </span>
                              </div>
                            )}
                            {event.correlation_id && (
                              <div>
                                <span className="text-gray-500">Correlation:</span>{' '}
                                <span className="text-gray-900 dark:text-gray-100">
                                  {event.correlation_id}
                                </span>
                              </div>
                            )}
                            <div className="mt-2">
                              <span className="text-gray-500">Payload:</span>
                              <pre className="mt-1 text-gray-900 dark:text-gray-100 overflow-x-auto">
                                {JSON.stringify(event.payload, null, 2)}
                              </pre>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// Helper function to render event summaries
function renderEventSummary(event: AgentEvent): React.ReactNode {
  const { type, payload } = event;

  switch (type) {
    case 'intent':
      return <span>User request: {payload.text || 'N/A'}</span>;

    case 'plan_generated':
      return renderPlanGeneratedSummary(payload);

    case 'plan_judged':
      return (
        <span>
          Score: {payload.overall_score ?? payload.score ?? 'N/A'}/100
          {payload.recommendations && payload.recommendations.length > 0 && (
            <> - {payload.recommendations[0]}</>
          )}
          {!payload.recommendations?.length && payload.notes && <> - {payload.notes}</>}
        </span>
      );

    case 'order_submitted':
      return (
        <span>
          {payload.side} {payload.qty} {payload.symbol} @ {payload.type || 'market'}
        </span>
      );

    case 'fill':
      return (
        <span>
          Filled: {payload.side} {payload.qty} {payload.symbol} @ $
          {payload.fill_price?.toFixed(2) || 'N/A'}
        </span>
      );

    case 'trade_blocked':
      return (
        <span className="text-red-600 dark:text-red-400">
          Blocked: {payload.reason || 'Unknown'} - {payload.detail || 'No details'}
        </span>
      );

    case 'llm_call':
      return (
        <span>
          {payload.model || 'LLM'} · in {payload.tokens_in ?? 'N/A'} / out {payload.tokens_out ?? 'N/A'} · {payload.duration_ms ?? 'N/A'}ms
        </span>
      );

    default:
      return <span>{JSON.stringify(payload).substring(0, 100)}...</span>;
  }
}

function renderPlanGeneratedSummary(payload: Record<string, any>): React.ReactNode {
  const planId = payload.plan_id || payload.strategy_id || payload.planId;
  const regime = payload.regime || payload.market_regime;
  const numTriggers = payload.num_triggers ?? (Array.isArray(payload.triggers) ? payload.triggers.length : undefined);
  const symbol = payload.symbol;
  const allowedSymbols = payload.allowed_symbols;
  const symbolLabel = symbol || (Array.isArray(allowedSymbols) ? allowedSymbols.join(', ') : 'unknown');
  const parts = [
    `Plan ${planId || 'N/A'}`,
    `symbol(s): ${symbolLabel}`,
  ];
  if (regime) {
    parts.push(`regime: ${regime}`);
  }
  if (numTriggers !== undefined) {
    parts.push(`triggers: ${numTriggers}`);
  }
  return <span>{parts.join(' · ')}</span>;
}
