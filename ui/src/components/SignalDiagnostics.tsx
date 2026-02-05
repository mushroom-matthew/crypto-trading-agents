import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { backtestAPI } from '../lib/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend, Cell,
} from 'recharts';

type Tab = 'triggers' | 'blocks' | 'judge' | 'samples';

export function SignalDiagnostics({ runId }: { runId: string }) {
  const [activeTab, setActiveTab] = useState<Tab>('triggers');
  const [selectedTrigger, setSelectedTrigger] = useState<string | null>(null);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Signal Diagnostics
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Hidden telemetry: per-trigger analytics, block analysis, judge history, and decision samples
        </p>
      </div>

      {/* Tab Bar */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 px-4">
        {([
          ['triggers', 'Trigger Analytics'],
          ['blocks', 'Block Analysis'],
          ['judge', 'Judge Timeline'],
          ['samples', 'Decision Samples'],
        ] as [Tab, string][]).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === key
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="p-4">
        {activeTab === 'triggers' && (
          <TriggerAnalyticsPanel
            runId={runId}
            onSelectTrigger={(tid) => {
              setSelectedTrigger(tid);
              setActiveTab('samples');
            }}
          />
        )}
        {activeTab === 'blocks' && <BlockAnalysisPanel runId={runId} />}
        {activeTab === 'judge' && <JudgeTimelinePanel runId={runId} />}
        {activeTab === 'samples' && (
          <DecisionSamplesPanel runId={runId} initialTrigger={selectedTrigger} />
        )}
      </div>
    </div>
  );
}

// ---- Trigger Analytics ----

function TriggerAnalyticsPanel({
  runId,
  onSelectTrigger,
}: {
  runId: string;
  onSelectTrigger: (tid: string) => void;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ['trigger-analytics', runId],
    queryFn: () => backtestAPI.getTriggerAnalytics(runId),
  });

  if (isLoading) return <div className="text-gray-500">Loading trigger analytics...</div>;
  if (!data) return <div className="text-gray-500">No trigger analytics available</div>;

  const triggers = (data.triggers || []).filter(
    (t: any) => t.trigger_id !== 'unknown'
  );

  // Chart data: executed vs blocked per trigger
  const chartData = triggers.map((t: any) => ({
    name: t.trigger_id.replace(/(btc_usd_|eth_usd_)/g, '').substring(0, 25),
    fullName: t.trigger_id,
    executed: t.daily_totals?.executed || 0,
    blocked: t.daily_totals?.blocked || 0,
    fired: t.eval_stats?.fired || 0,
  }));

  return (
    <div className="space-y-6">
      {/* Execution vs Block chart */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Trigger Execution vs Blocks
        </h4>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 150 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={145} />
            <Tooltip />
            <Legend />
            <Bar dataKey="executed" fill="#22c55e" name="Executed" />
            <Bar dataKey="blocked" fill="#ef4444" name="Blocked" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Trigger table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-gray-500 dark:text-gray-400 border-b dark:border-gray-700">
              <th className="pb-2 pr-4">Trigger</th>
              <th className="pb-2 pr-4 text-right">Evaluated</th>
              <th className="pb-2 pr-4 text-right">Fired</th>
              <th className="pb-2 pr-4 text-right">Executed</th>
              <th className="pb-2 pr-4 text-right">Blocked</th>
              <th className="pb-2 pr-4 text-right">Fire Rate</th>
              <th className="pb-2 pr-4">Block Reasons</th>
            </tr>
          </thead>
          <tbody>
            {triggers.map((t: any) => {
              const evals = t.eval_stats?.evaluated || 0;
              const fired = t.eval_stats?.fired || 0;
              const executed = t.daily_totals?.executed || 0;
              const blocked = t.daily_totals?.blocked || 0;
              const fireRate = evals > 0 ? ((fired / evals) * 100).toFixed(1) : '-';
              const reasons = t.daily_totals?.blocked_reasons || {};

              return (
                <tr
                  key={t.trigger_id}
                  className="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750 cursor-pointer"
                  onClick={() => onSelectTrigger(t.trigger_id)}
                >
                  <td className="py-2 pr-4 font-mono text-xs text-blue-600 dark:text-blue-400">
                    {t.trigger_id}
                  </td>
                  <td className="py-2 pr-4 text-right">{evals}</td>
                  <td className="py-2 pr-4 text-right">{fired}</td>
                  <td className="py-2 pr-4 text-right font-medium text-green-600">{executed}</td>
                  <td className="py-2 pr-4 text-right font-medium text-red-500">{blocked}</td>
                  <td className="py-2 pr-4 text-right">{fireRate}%</td>
                  <td className="py-2 pr-4">
                    {Object.entries(reasons).map(([reason, count]) => (
                      <span
                        key={reason}
                        className="inline-block bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-xs px-1.5 py-0.5 rounded mr-1 mb-0.5"
                      >
                        {reason}: {count as number}
                      </span>
                    ))}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---- Block Analysis ----

function BlockAnalysisPanel({ runId }: { runId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['block-analysis', runId],
    queryFn: () => backtestAPI.getBlockAnalysis(runId),
  });

  if (isLoading) return <div className="text-gray-500">Loading block analysis...</div>;
  if (!data) return <div className="text-gray-500">No block data available</div>;

  const REASON_COLORS: Record<string, string> = {
    expression_error: '#ef4444',
    exit_binding_mismatch: '#f59e0b',
    min_flat: '#8b5cf6',
    min_hold: '#3b82f6',
    risk: '#dc2626',
    risk_budget: '#ea580c',
  };

  const reasonChartData = Object.entries(data.reason_totals || {}).map(([reason, count]) => ({
    name: reason,
    count: count as number,
    fill: REASON_COLORS[reason] || '#6b7280',
  }));

  const dailyData = (data.daily_blocks || []).map((d: any) => ({
    date: d.date?.substring(5),
    blocks: d.total_blocks,
    trades: d.executed_trades,
    execRate: d.execution_rate ? (d.execution_rate * 100).toFixed(0) : null,
  }));

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <SummaryCard label="Total Blocks" value={data.total_blocks} color="red" />
        <SummaryCard
          label="Top Reason"
          value={Object.keys(data.reason_totals || {})[0] || '-'}
          color="amber"
        />
        <SummaryCard
          label="Most Blocked Trigger"
          value={(Object.keys(data.trigger_blocks || {})[0] || '-').replace(/(btc_usd_|eth_usd_)/, '')}
          color="orange"
        />
        <SummaryCard
          label="Block Details Available"
          value={(data.block_details || []).length}
          color="gray"
        />
      </div>

      {/* Block Reason Distribution */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Block Reasons
        </h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={reasonChartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" name="Blocks">
              {reasonChartData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Daily Blocks vs Trades */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Daily Blocks vs Executed Trades
        </h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={dailyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="trades" fill="#22c55e" name="Executed" />
            <Bar dataKey="blocks" fill="#ef4444" name="Blocked" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Block Details Table */}
      {(data.block_details || []).length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Recent Block Events (first 50)
          </h4>
          <div className="overflow-x-auto max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-white dark:bg-gray-800">
                <tr className="text-left text-gray-500 dark:text-gray-400 border-b dark:border-gray-700">
                  <th className="pb-2 pr-3">Time</th>
                  <th className="pb-2 pr-3">Symbol</th>
                  <th className="pb-2 pr-3">Trigger</th>
                  <th className="pb-2 pr-3">Reason</th>
                  <th className="pb-2">Detail</th>
                </tr>
              </thead>
              <tbody>
                {data.block_details.slice(0, 50).map((b: any, i: number) => (
                  <tr key={i} className="border-b dark:border-gray-700">
                    <td className="py-1.5 pr-3 font-mono whitespace-nowrap">
                      {b.timestamp?.substring(5, 16)}
                    </td>
                    <td className="py-1.5 pr-3">{b.symbol}</td>
                    <td className="py-1.5 pr-3 font-mono text-blue-600 dark:text-blue-400">
                      {(b.trigger_id || '').replace(/(btc_usd_|eth_usd_)/, '')}
                    </td>
                    <td className="py-1.5 pr-3">
                      <span className="bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 px-1.5 py-0.5 rounded">
                        {b.reason}
                      </span>
                    </td>
                    <td className="py-1.5 text-gray-600 dark:text-gray-400 max-w-md truncate">
                      {b.detail}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ---- Judge Timeline ----

function JudgeTimelinePanel({ runId }: { runId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['judge-history', runId],
    queryFn: () => backtestAPI.getJudgeHistory(runId),
  });

  if (isLoading) return <div className="text-gray-500">Loading judge history...</div>;
  if (!data) return <div className="text-gray-500">No judge history available</div>;

  const evals = data.evaluations || [];

  const chartData = evals.map((e: any) => ({
    time: e.timestamp?.substring(5, 16),
    score: e.score,
    replan: e.should_replan ? 100 : null,
    reason: e.trigger_reason,
  }));

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="grid grid-cols-3 gap-4">
        <SummaryCard label="Total Evaluations" value={data.total_evaluations} color="blue" />
        <SummaryCard label="Replans Triggered" value={data.total_replans} color="amber" />
        <SummaryCard
          label="Avg Score"
          value={
            evals.length > 0
              ? (evals.reduce((s: number, e: any) => s + (e.score || 0), 0) / evals.length).toFixed(1)
              : '-'
          }
          color="green"
        />
      </div>

      {/* Score Timeline */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Judge Score Over Time
        </h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
            <YAxis domain={[0, 100]} />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const d = payload[0]?.payload;
                return (
                  <div className="bg-white dark:bg-gray-800 border rounded p-2 text-xs shadow">
                    <div>Time: {d.time}</div>
                    <div>Score: {d.score}</div>
                    <div>Reason: {d.reason}</div>
                    {d.replan && <div className="text-red-500 font-bold">REPLAN TRIGGERED</div>}
                  </div>
                );
              }}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 2 }}
              name="Score"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Evaluation Log */}
      <div className="overflow-x-auto max-h-72 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-white dark:bg-gray-800">
            <tr className="text-left text-gray-500 dark:text-gray-400 border-b dark:border-gray-700">
              <th className="pb-2 pr-3">Time</th>
              <th className="pb-2 pr-3 text-right">Score</th>
              <th className="pb-2 pr-3">Trigger</th>
              <th className="pb-2 pr-3">Replan</th>
              <th className="pb-2">Notes</th>
            </tr>
          </thead>
          <tbody>
            {evals.map((e: any, i: number) => (
              <tr key={i} className="border-b dark:border-gray-700">
                <td className="py-1.5 pr-3 font-mono whitespace-nowrap">
                  {e.timestamp?.substring(5, 16)}
                </td>
                <td className="py-1.5 pr-3 text-right font-medium">
                  <span
                    className={
                      e.score >= 60
                        ? 'text-green-600'
                        : e.score >= 40
                        ? 'text-yellow-600'
                        : 'text-red-600'
                    }
                  >
                    {e.score?.toFixed(1)}
                  </span>
                </td>
                <td className="py-1.5 pr-3">{e.trigger_reason}</td>
                <td className="py-1.5 pr-3">
                  {e.should_replan ? (
                    <span className="bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300 px-1.5 py-0.5 rounded">
                      Yes
                    </span>
                  ) : (
                    <span className="text-gray-400">No</span>
                  )}
                </td>
                <td className="py-1.5 text-gray-600 dark:text-gray-400 max-w-sm truncate">
                  {e.feedback?.notes?.substring(0, 100)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---- Decision Samples ----

function DecisionSamplesPanel({
  runId,
  initialTrigger,
}: {
  runId: string;
  initialTrigger: string | null;
}) {
  const [triggerId, setTriggerId] = useState(initialTrigger || '');
  const [resultFilter, setResultFilter] = useState<string>('all');

  const { data, isLoading } = useQuery({
    queryKey: ['trigger-samples', runId, triggerId, resultFilter],
    queryFn: () =>
      backtestAPI.getTriggerSamples(runId, {
        trigger_id: triggerId || undefined,
        result: resultFilter === 'all' ? undefined : resultFilter === 'true',
        limit: 200,
      }),
  });

  const samples = data?.samples || [];

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex gap-3 items-center">
        <input
          type="text"
          placeholder="Filter by trigger ID..."
          value={triggerId}
          onChange={(e) => setTriggerId(e.target.value)}
          className="flex-1 px-3 py-1.5 text-sm border rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
        />
        <select
          value={resultFilter}
          onChange={(e) => setResultFilter(e.target.value)}
          className="px-3 py-1.5 text-sm border rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
        >
          <option value="all">All Results</option>
          <option value="true">Fired (true)</option>
          <option value="false">Not Fired (false)</option>
        </select>
        <span className="text-sm text-gray-500">
          {isLoading ? 'Loading...' : `${data?.total || 0} total, showing ${samples.length}`}
        </span>
      </div>

      {/* Samples */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {samples.map((s: any, i: number) => (
          <div
            key={i}
            className={`border rounded p-3 text-xs ${
              s.result
                ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/10'
                : 'border-gray-200 dark:border-gray-700'
            }`}
          >
            <div className="flex justify-between items-start mb-2">
              <div>
                <span className="font-mono text-blue-600 dark:text-blue-400 font-medium">
                  {s.trigger_id}
                </span>
                <span className="ml-2 text-gray-500">{s.rule_type}</span>
                <span className="ml-2 text-gray-400">{s.symbol}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-gray-500">
                  {s.timestamp?.substring(5, 16)}
                </span>
                <span
                  className={`px-2 py-0.5 rounded font-medium ${
                    s.result
                      ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                      : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                  }`}
                >
                  {s.result ? 'FIRED' : 'not fired'}
                </span>
              </div>
            </div>

            {/* Expression */}
            <div className="font-mono text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-900/50 px-2 py-1 rounded mb-2">
              {s.expression}
            </div>

            {/* Sub-results */}
            {s.sub_results && (
              <div className="space-y-0.5">
                {s.sub_results.map(([desc, result]: [string, boolean], j: number) => (
                  <div key={j} className="flex items-center gap-2">
                    <span
                      className={`w-4 h-4 flex items-center justify-center rounded-full text-[10px] font-bold ${
                        result
                          ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                          : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                      }`}
                    >
                      {result ? '\u2713' : '\u2717'}
                    </span>
                    <span className="font-mono text-gray-600 dark:text-gray-400">{desc}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Context Values */}
            {s.context_values && Object.keys(s.context_values).length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {Object.entries(s.context_values).map(([key, val]) => (
                  <span
                    key={key}
                    className="bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 px-1.5 py-0.5 rounded"
                  >
                    {key}={typeof val === 'number' ? (val as number).toFixed(2) : String(val)}
                  </span>
                ))}
              </div>
            )}

            {s.error && (
              <div className="mt-2 text-red-600 dark:text-red-400">Error: {s.error}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---- Helpers ----

function SummaryCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color: string;
}) {
  const colorClasses: Record<string, string> = {
    red: 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300',
    green: 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300',
    blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300',
    amber: 'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300',
    orange: 'bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300',
    gray: 'bg-gray-50 dark:bg-gray-900/20 text-gray-700 dark:text-gray-300',
  };

  return (
    <div className={`rounded-lg p-3 ${colorClasses[color] || colorClasses.gray}`}>
      <div className="text-xs font-medium opacity-75">{label}</div>
      <div className="text-lg font-bold mt-0.5">{value}</div>
    </div>
  );
}
