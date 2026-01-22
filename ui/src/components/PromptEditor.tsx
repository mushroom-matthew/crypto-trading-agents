import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Save, RotateCcw, ChevronDown, ChevronUp, FileText, AlertCircle, Check, History, Clock } from 'lucide-react';
import { promptsAPI } from '../lib/api';
import { cn, formatDateTime } from '../lib/utils';

type PromptType = 'strategist' | 'judge';

interface PromptEditorProps {
  defaultExpanded?: boolean;
}

export function PromptEditor({ defaultExpanded = false }: PromptEditorProps) {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [selectedPrompt, setSelectedPrompt] = useState<PromptType>('strategist');
  const [editedContent, setEditedContent] = useState<string>('');
  const [hasChanges, setHasChanges] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [showVersions, setShowVersions] = useState(false);

  // Fetch the selected prompt
  const { data: prompt, isLoading, error, isFetched } = useQuery({
    queryKey: ['prompt', selectedPrompt],
    queryFn: () => promptsAPI.get(selectedPrompt),
    enabled: expanded,
    staleTime: 0, // Always fetch fresh data
  });

  // Fetch versions
  const { data: versionsData, refetch: refetchVersions } = useQuery({
    queryKey: ['prompt-versions', selectedPrompt],
    queryFn: () => promptsAPI.listVersions(selectedPrompt),
    enabled: expanded && showVersions,
  });

  // Update local state when prompt data changes
  useEffect(() => {
    if (prompt?.content !== undefined) {
      setEditedContent(prompt.content);
      setHasChanges(false);
    }
  }, [prompt?.content]);

  // Update prompt mutation
  const updateMutation = useMutation({
    mutationFn: ({ name, content }: { name: string; content: string }) =>
      promptsAPI.update(name, content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompt', selectedPrompt] });
      queryClient.invalidateQueries({ queryKey: ['prompt-versions', selectedPrompt] });
      setHasChanges(false);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2000);
    },
  });

  // Reset prompt mutation
  const resetMutation = useMutation({
    mutationFn: (name: string) => promptsAPI.reset(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompt', selectedPrompt] });
      queryClient.invalidateQueries({ queryKey: ['prompt-versions', selectedPrompt] });
      setHasChanges(false);
    },
  });

  // Restore version mutation
  const restoreMutation = useMutation({
    mutationFn: ({ name, versionId }: { name: string; versionId: string }) =>
      promptsAPI.restoreVersion(name, versionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompt', selectedPrompt] });
      queryClient.invalidateQueries({ queryKey: ['prompt-versions', selectedPrompt] });
      setHasChanges(false);
      setShowVersions(false);
    },
  });

  const handleContentChange = (value: string) => {
    setEditedContent(value);
    setHasChanges(value !== prompt?.content);
  };

  const handleSave = () => {
    updateMutation.mutate({ name: selectedPrompt, content: editedContent });
  };

  const handleReset = () => {
    if (selectedPrompt === 'strategist') {
      // Can't reset strategist, just reload from server
      queryClient.invalidateQueries({ queryKey: ['prompt', selectedPrompt] });
    } else {
      resetMutation.mutate(selectedPrompt);
    }
  };

  const handlePromptSwitch = (promptType: PromptType) => {
    if (hasChanges) {
      const confirm = window.confirm('You have unsaved changes. Switch anyway?');
      if (!confirm) return;
    }
    setSelectedPrompt(promptType);
    setHasChanges(false);
    setShowVersions(false);
  };

  const handleRestoreVersion = (versionId: string) => {
    if (window.confirm(`Restore to version ${versionId}? Current version will be backed up.`)) {
      restoreMutation.mutate({ name: selectedPrompt, versionId });
    }
  };

  const versions = versionsData?.versions || [];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
      {/* Header - Collapsible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <FileText className="w-5 h-5 text-blue-500" />
          <div className="text-left">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">LLM Prompt Editor</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Edit strategist and judge prompts for backtesting
            </p>
          </div>
        </div>
        {expanded ? (
          <ChevronUp className="w-5 h-5 text-gray-500" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-500" />
        )}
      </button>

      {/* Expandable Content */}
      {expanded && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4 space-y-4">
          {/* Toggle Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => handlePromptSwitch('strategist')}
              className={cn(
                'flex-1 py-2 px-4 rounded-lg font-medium transition-colors',
                selectedPrompt === 'strategist'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              )}
            >
              Strategist Prompt
            </button>
            <button
              onClick={() => handlePromptSwitch('judge')}
              className={cn(
                'flex-1 py-2 px-4 rounded-lg font-medium transition-colors',
                selectedPrompt === 'judge'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              )}
            >
              Judge Prompt
            </button>
          </div>

          {/* Description */}
          <div className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg">
            {selectedPrompt === 'strategist' ? (
              <p>
                <strong>Strategist Prompt:</strong> Controls how the LLM generates trading strategies,
                trigger conditions, and risk parameters during backtesting.
              </p>
            ) : (
              <p>
                <strong>Judge Prompt:</strong> Controls how the LLM evaluates trading performance,
                identifies issues, and provides constraints for strategy improvement.
              </p>
            )}
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
              <span className="ml-3 text-gray-500">Loading prompt...</span>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="flex items-center gap-2 p-4 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-lg">
              <AlertCircle className="w-5 h-5" />
              <p>Failed to load prompt: {(error as Error).message}</p>
            </div>
          )}

          {/* Editor */}
          {isFetched && !isLoading && !error && (
            <>
              <div className="relative">
                <textarea
                  value={editedContent}
                  onChange={(e) => handleContentChange(e.target.value)}
                  className={cn(
                    'w-full h-96 p-4 font-mono text-sm rounded-lg border resize-y',
                    'bg-gray-50 dark:bg-gray-900',
                    'text-gray-900 dark:text-gray-100',
                    'border-gray-300 dark:border-gray-600',
                    'focus:ring-2 focus:ring-blue-500 focus:border-transparent',
                    'scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-200'
                  )}
                  placeholder="Enter prompt content..."
                />
                {hasChanges && (
                  <div className="absolute top-2 right-2 px-2 py-1 bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-400 text-xs rounded">
                    Unsaved changes
                  </div>
                )}
              </div>

              {/* Character Count & File Path */}
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>{editedContent.length.toLocaleString()} characters</span>
                {prompt?.file_path && (
                  <span className="font-mono truncate max-w-md">{prompt.file_path}</span>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handleSave}
                  disabled={!hasChanges || updateMutation.isPending}
                  className={cn(
                    'flex items-center gap-2 py-2 px-4 rounded-lg font-medium transition-colors',
                    hasChanges && !updateMutation.isPending
                      ? 'bg-green-600 hover:bg-green-700 text-white'
                      : 'bg-gray-300 dark:bg-gray-600 text-gray-500 cursor-not-allowed'
                  )}
                >
                  {saveSuccess ? (
                    <>
                      <Check className="w-4 h-4" />
                      Saved!
                    </>
                  ) : updateMutation.isPending ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      Save Changes
                    </>
                  )}
                </button>

                <button
                  onClick={() => {
                    setShowVersions(!showVersions);
                    if (!showVersions) {
                      refetchVersions();
                    }
                  }}
                  className={cn(
                    'flex items-center gap-2 py-2 px-4 rounded-lg font-medium transition-colors',
                    showVersions
                      ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-400'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  )}
                >
                  <History className="w-4 h-4" />
                  Version History
                </button>

                {selectedPrompt === 'judge' && (
                  <button
                    onClick={handleReset}
                    disabled={resetMutation.isPending}
                    className={cn(
                      'flex items-center gap-2 py-2 px-4 rounded-lg font-medium transition-colors',
                      'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
                      'hover:bg-gray-200 dark:hover:bg-gray-600'
                    )}
                  >
                    {resetMutation.isPending ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-500" />
                        Resetting...
                      </>
                    ) : (
                      <>
                        <RotateCcw className="w-4 h-4" />
                        Reset to Default
                      </>
                    )}
                  </button>
                )}

                <button
                  onClick={() => {
                    setEditedContent(prompt?.content || '');
                    setHasChanges(false);
                  }}
                  disabled={!hasChanges}
                  className={cn(
                    'flex items-center gap-2 py-2 px-4 rounded-lg font-medium transition-colors',
                    hasChanges
                      ? 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                      : 'bg-gray-200 dark:bg-gray-800 text-gray-400 cursor-not-allowed'
                  )}
                >
                  Discard Changes
                </button>
              </div>

              {/* Version History Panel */}
              {showVersions && (
                <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                  <div className="bg-gray-50 dark:bg-gray-700/50 px-4 py-2 border-b border-gray-200 dark:border-gray-700">
                    <h4 className="font-medium text-sm text-gray-700 dark:text-gray-300 flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      Previous Versions
                    </h4>
                  </div>
                  <div className="max-h-48 overflow-y-auto">
                    {versions.length === 0 ? (
                      <div className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                        No previous versions yet. Versions are created automatically when you save changes.
                      </div>
                    ) : (
                      <div className="divide-y divide-gray-200 dark:divide-gray-700">
                        {versions.map((version) => (
                          <div
                            key={version.version_id}
                            className="px-4 py-2 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/30"
                          >
                            <div className="text-sm">
                              <span className="font-mono text-gray-600 dark:text-gray-400">
                                {version.version_id}
                              </span>
                              <span className="ml-3 text-gray-500 dark:text-gray-500">
                                {formatDateTime(version.timestamp)}
                              </span>
                              <span className="ml-3 text-gray-400 dark:text-gray-600 text-xs">
                                {(version.size_bytes / 1024).toFixed(1)} KB
                              </span>
                            </div>
                            <button
                              onClick={() => handleRestoreVersion(version.version_id)}
                              disabled={restoreMutation.isPending}
                              className="px-3 py-1 text-xs font-medium bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-400 rounded hover:bg-blue-200 dark:hover:bg-blue-900"
                            >
                              Restore
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Mutation Errors */}
              {updateMutation.isError && (
                <div className="flex items-center gap-2 p-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-lg text-sm">
                  <AlertCircle className="w-4 h-4" />
                  Failed to save: {(updateMutation.error as Error).message}
                </div>
              )}

              {resetMutation.isError && (
                <div className="flex items-center gap-2 p-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-lg text-sm">
                  <AlertCircle className="w-4 h-4" />
                  Failed to reset: {(resetMutation.error as Error).message}
                </div>
              )}

              {restoreMutation.isError && (
                <div className="flex items-center gap-2 p-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-lg text-sm">
                  <AlertCircle className="w-4 h-4" />
                  Failed to restore: {(restoreMutation.error as Error).message}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
