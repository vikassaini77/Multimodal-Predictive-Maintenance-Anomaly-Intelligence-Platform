import React, { useState, useEffect } from 'react';
import { Calendar, Clock, MapPin, Plus, Trash2 } from 'lucide-react';

interface MaintenanceWindow {
  id: string;
  zone: string;
  day_of_week: number;
  start_time: string;
  end_time: string;
}

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const ZONES = ['Milling Zone', 'Stamping Zone', 'Welding Zone', 'General Zone'];

export function MaintenanceScheduler() {
  const [windows, setWindows] = useState<MaintenanceWindow[]>([]);
  const [loading, setLoading] = useState(true);
  
  const [zone, setZone] = useState(ZONES[0]);
  const [dayOfWeek, setDayOfWeek] = useState(0);
  const [startTime, setStartTime] = useState("02:00");
  const [endTime, setEndTime] = useState("04:00");

  useEffect(() => {
    fetchWindows();
  }, []);

  const fetchWindows = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/maintenance/windows');
      if (res.ok) {
        const data = await res.json();
        setWindows(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/maintenance/windows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          zone,
          day_of_week: dayOfWeek,
          start_time: startTime,
          end_time: endTime
        })
      });
      if (res.ok) {
        const newWindow = await res.json();
        setWindows(prev => [...prev, newWindow]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      const res = await fetch(`/api/maintenance/windows/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setWindows(prev => prev.filter(w => w.id !== id));
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl flex flex-col h-full">
      <h2 className="text-xl font-bold mb-4 text-slate-100 flex items-center gap-2">
        <Calendar size={20} className="text-primary" />
        Maintenance Windows
      </h2>

      <form onSubmit={handleAdd} className="flex gap-2 mb-4 text-sm">
        <select 
          className="bg-slate-800 border border-slate-700 text-slate-200 rounded p-1.5 focus:border-primary outline-none"
          value={zone} onChange={(e) => setZone(e.target.value)}
        >
          {ZONES.map(z => <option key={z} value={z}>{z}</option>)}
        </select>
        <select 
          className="bg-slate-800 border border-slate-700 text-slate-200 rounded p-1.5 focus:border-primary outline-none"
          value={dayOfWeek} onChange={(e) => setDayOfWeek(parseInt(e.target.value))}
        >
          {DAYS.map((d, i) => <option key={i} value={i}>{d}</option>)}
        </select>
        <input 
          type="time" 
          required 
          className="bg-slate-800 border border-slate-700 text-slate-200 rounded p-1.5 focus:border-primary outline-none"
          value={startTime} onChange={(e) => setStartTime(e.target.value)}
        />
        <span className="text-slate-400 self-center">-</span>
        <input 
          type="time" 
          required 
          className="bg-slate-800 border border-slate-700 text-slate-200 rounded p-1.5 focus:border-primary outline-none"
          value={endTime} onChange={(e) => setEndTime(e.target.value)}
        />
        <button type="submit" className="bg-primary text-slate-900 rounded px-3 py-1.5 font-bold hover:bg-primary/90 flex items-center gap-1">
          <Plus size={16} /> Add
        </button>
      </form>

      <div className="flex-1 overflow-y-auto space-y-2 custom-scrollbar">
        {loading ? (
          <div className="text-slate-400 text-sm">Loading schedules...</div>
        ) : windows.length === 0 ? (
          <div className="text-slate-400 text-sm">No maintenance windows configured.</div>
        ) : (
          windows.map(w => (
            <div key={w.id} className="flex items-center justify-between bg-slate-800/50 border border-slate-700 p-2 rounded text-sm">
              <div className="flex items-center gap-4 text-slate-300">
                <span className="flex items-center gap-1 font-semibold text-accent"><MapPin size={14}/> {w.zone}</span>
                <span className="flex items-center gap-1"><Calendar size={14}/> {DAYS[w.day_of_week]}</span>
                <span className="flex items-center gap-1"><Clock size={14}/> {w.start_time.substring(0,5)} - {w.end_time.substring(0,5)}</span>
              </div>
              <button onClick={() => handleDelete(w.id)} className="text-red-400 hover:text-red-300">
                <Trash2 size={16} />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
