import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { walletsAPI } from '../lib/api'

interface Wallet {
  wallet_id: number
  name: string
  currency: string | null
  ledger_balance: number
  coinbase_balance: number | null
  drift: number | null
  tradeable_fraction: number
  type: string
}

interface DriftRecord {
  wallet_id: number
  wallet_name: string
  currency: string
  ledger_balance: number
  coinbase_balance: number
  drift: number
  within_threshold: boolean
}

interface ReconciliationReport {
  timestamp: string
  total_wallets: number
  drifts_detected: number
  drifts_within_threshold: number
  drifts_exceeding_threshold: number
  records: DriftRecord[]
}

export default function WalletReconciliation() {
  const [threshold, setThreshold] = useState(0.0001)
  const queryClient = useQueryClient()

  // Query wallets
  const { data: wallets = [], isLoading: walletsLoading } = useQuery({
    queryKey: ['wallets'],
    queryFn: () => walletsAPI.list(),
  })

  // Reconciliation mutation
  const reconcileMutation = useMutation({
    mutationFn: (threshold: number) => walletsAPI.reconcile({ threshold }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wallets'] })
    }
  })

  const handleReconcile = () => {
    reconcileMutation.mutate(threshold)
  }

  const lastReport = reconcileMutation.data as ReconciliationReport | undefined

  return (
    <div className="space-y-6">
      <div className="bg-white shadow-sm rounded-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Wallet Reconciliation</h2>
        <p className="text-gray-600 mb-6">
          Compare ledger balances with Coinbase exchange balances to detect drift and ensure data integrity.
        </p>

        {/* Reconciliation Controls */}
        <div className="flex items-center gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Drift Threshold:</label>
            <input
              type="number"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              step="0.0001"
              className="w-32 px-3 py-2 border border-gray-300 rounded-md text-sm"
            />
          </div>
          <button
            onClick={handleReconcile}
            disabled={reconcileMutation.isPending}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-sm font-medium"
          >
            {reconcileMutation.isPending ? 'Running...' : 'Run Reconciliation'}
          </button>
        </div>

        {/* Last Reconciliation Report */}
        {lastReport && (
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="font-semibold text-blue-900 mb-2">Last Reconciliation Report</h3>
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-600">Timestamp</div>
                <div className="font-medium">{new Date(lastReport.timestamp).toLocaleString()}</div>
              </div>
              <div>
                <div className="text-gray-600">Total Wallets</div>
                <div className="font-medium">{lastReport.total_wallets}</div>
              </div>
              <div>
                <div className="text-gray-600">Drifts Detected</div>
                <div className="font-medium text-yellow-600">{lastReport.drifts_detected}</div>
              </div>
              <div>
                <div className="text-gray-600">Exceeding Threshold</div>
                <div className={`font-medium ${lastReport.drifts_exceeding_threshold > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {lastReport.drifts_exceeding_threshold}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {reconcileMutation.isError && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <h3 className="font-semibold text-red-900 mb-2">Reconciliation Failed</h3>
            <p className="text-sm text-red-700">{(reconcileMutation.error as Error).message}</p>
          </div>
        )}
      </div>

      {/* Wallets Table */}
      <div className="bg-white shadow-sm rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Wallets</h3>
        {walletsLoading ? (
          <div className="text-center py-8 text-gray-500">Loading wallets...</div>
        ) : wallets.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No wallets found. Run <code className="bg-gray-100 px-2 py-1 rounded">uv run python -m app.cli.main ledger seed-from-coinbase</code> to populate wallets.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Wallet
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Currency
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Ledger Balance
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Coinbase Balance
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Drift
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Tradeable %
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {wallets.map((wallet: Wallet) => (
                  <tr key={wallet.wallet_id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {wallet.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {wallet.currency || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900">
                      {wallet.ledger_balance.toFixed(8)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-500">
                      {wallet.coinbase_balance !== null ? wallet.coinbase_balance.toFixed(8) : '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right">
                      {wallet.drift !== null ? (
                        <span className={wallet.drift === 0 ? 'text-green-600' : 'text-yellow-600'}>
                          {wallet.drift.toFixed(8)}
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {(wallet.tradeable_fraction * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {wallet.type}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Drift Details Table */}
      {lastReport && lastReport.records.length > 0 && (
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Drift Details</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Wallet
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Currency
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Ledger
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Coinbase
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Drift
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {lastReport.records.map((record: DriftRecord, idx: number) => (
                  <tr key={idx} className={!record.within_threshold ? 'bg-red-50' : ''}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {record.wallet_name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {record.currency}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900">
                      {record.ledger_balance.toFixed(8)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900">
                      {record.coinbase_balance.toFixed(8)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-medium">
                      <span className={Math.abs(record.drift) === 0 ? 'text-green-600' : Math.abs(record.drift) > threshold ? 'text-red-600' : 'text-yellow-600'}>
                        {record.drift.toFixed(8)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {record.within_threshold ? (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          OK
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          DRIFT!
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
