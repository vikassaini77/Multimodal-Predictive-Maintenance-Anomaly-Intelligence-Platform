import React, { useState } from 'react';
import { Activity, Lock, User } from 'lucide-react';
import clsx from 'clsx';

interface LoginProps {
  onSuccess: () => void;
}

export function Login({ onSuccess }: LoginProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      
      if (!res.ok) {
        throw new Error('Invalid credentials');
      }
      
      onSuccess();
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-md bg-surface p-8 rounded-2xl border border-slate-700 shadow-2xl">
        <div className="flex flex-col items-center mb-8">
          <div className="bg-primary/20 p-3 rounded-xl border border-primary/30 mb-4">
            <Activity className="text-primary" size={32} />
          </div>
          <h1 className="text-2xl font-bold text-slate-100">IndustrialMind AI</h1>
          <p className="text-slate-400 text-sm mt-1">Authenticate to access operations dashboard</p>
        </div>
        
        {error && (
          <div className="bg-red-900/50 border border-red-500/50 text-red-200 text-sm p-3 rounded-lg mb-6 text-center">
            {error}
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Username</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <User className="text-slate-500" size={16} />
              </div>
              <input
                type="text"
                required
                className="w-full bg-background border border-slate-700 rounded-lg py-2.5 pl-10 pr-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-colors"
                placeholder="Enter username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
          </div>
          
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Password</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="text-slate-500" size={16} />
              </div>
              <input
                type="password"
                required
                className="w-full bg-background border border-slate-700 rounded-lg py-2.5 pl-10 pr-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-colors"
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>
          
          <button
            type="submit"
            disabled={loading}
            className={clsx(
              "w-full mt-6 py-2.5 px-4 rounded-lg font-bold text-slate-900 transition-all shadow-lg",
              loading 
                ? "bg-primary/50 cursor-not-allowed" 
                : "bg-primary hover:bg-primary/90 hover:shadow-primary/20"
            )}
          >
            {loading ? "Authenticating..." : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
}
