import React, { useEffect, useState } from 'react';
import { AlertTriangle, RefreshCcw, XCircle } from 'lucide-react';

interface FailedJob {
  task_id: string;
  failed_at: number;
  args: any[];
  kwargs: any;
  exc: string;
  traceback: string | null;
}

export function FailedJobsPanel() {
  const [jobs, setJobs] = useState<FailedJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);

  const fetchJobs = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/graph/jobs/failed');
      const data = await res.json();
      if (data.status === 'success') {
        setJobs(data.failed_jobs);
      }
    } catch (err) {
      console.error("Failed to fetch DLQ:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();

    // Setup an event listener for custom "refresh-dlq" events fired by toasts
    const handleRefresh = () => fetchJobs();
    window.addEventListener('refresh-dlq', handleRefresh);
    return () => window.removeEventListener('refresh-dlq', handleRefresh);
  }, []);

  if (loading) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl flex items-center justify-center text-slate-400 h-32 mt-6">
        <RefreshCcw className="animate-spin" size={24} />
      </div>
    );
  }

  if (jobs.length === 0) {
    return null; // Don't show the panel if there are no failed jobs
  }

  return (
    <div className="bg-red-900/10 rounded-lg p-4 border border-red-900/50 shadow-xl mt-6 animate-fade-in">
      <div className="flex items-center justify-between mb-4 pb-2 border-b border-red-900/30">
        <h2 className="text-lg font-bold text-red-400 flex items-center gap-2">
          <AlertTriangle size={20} />
          Dead Letter Queue ({jobs.length})
        </h2>
        <button onClick={fetchJobs} className="text-slate-400 hover:text-white transition-colors">
          <RefreshCcw size={16} />
        </button>
      </div>

      <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
        {jobs.map((job, idx) => (
          <div key={idx} className="bg-background rounded border border-red-900/30 p-3">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm font-mono text-slate-300">Job: {job.task_id}</p>
                <p className="text-xs text-slate-500">{new Date(job.failed_at * 1000).toLocaleString()}</p>
              </div>
              <button 
                onClick={() => setExpandedJob(expandedJob === job.task_id ? null : job.task_id)}
                className="text-xs bg-red-900/30 hover:bg-red-900/50 text-red-200 px-2 py-1 rounded transition-colors"
              >
                {expandedJob === job.task_id ? 'Hide Traceback' : 'View Error'}
              </button>
            </div>
            
            <div className="mt-2 text-sm text-red-400 flex items-start gap-2">
              <XCircle size={16} className="mt-0.5 shrink-0" />
              <span className="font-mono text-xs overflow-hidden text-ellipsis">{job.exc}</span>
            </div>

            {expandedJob === job.task_id && job.traceback && (
              <div className="mt-3 bg-black/50 p-2 rounded overflow-x-auto">
                <pre className="text-[10px] text-red-300 font-mono leading-relaxed">
                  {job.traceback}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
