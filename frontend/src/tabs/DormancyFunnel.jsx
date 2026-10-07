import { useState, useEffect, useCallback } from 'react';
import { TrendingUp, TrendingDown, DollarSign, ShieldAlert, Users, Zap, Lightbulb } from 'lucide-react';
import { simulateGrowth, getOverview } from '../lib/api';

function KPI({ icon: Icon, label, value, sub, color = 'text-white' }) {
  return (
    <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
      <div className="flex items-center gap-2 mb-2">
        <Icon size={16} className="text-slate-400" />
        <span className="text-xs text-slate-400 uppercase tracking-wide">{label}</span>
      </div>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  );
}

export default function DormancyFunnel() {
  const [overview, setOverview] = useState(null);
  const [budget, setBudget] = useState(10000);
  const [fatigueCap, setFatigueCap] = useState(70);
  const [upliftCut, setUpliftCut] = useState(0.02);
  const [stage, setStage] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { getOverview().then(setOverview).catch(console.error); }, []);

  const simulate = useCallback(async () => {
    setLoading(true);
    try {
      const params = { budget_bdt: budget, max_fatigue_cap: fatigueCap, min_uplift_cutoff: upliftCut };
      if (stage) params.lifecycle_stage = stage;
      const r = await simulateGrowth(params);
      setResult(r);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, [budget, fatigueCap, upliftCut, stage]);

  useEffect(() => { simulate(); }, [simulate]);

  const funnel = overview?.funnel || {};
  const stages = ['Sign-Up Drop-off', 'Payday Cash-Outer', 'One-Hit Wonder', 'At-Risk Churner'];
  const stageColors = ['#EF4444', '#F59E0B', '#8B5CF6', '#3B82F6'];

  const ai = result?.activate_ai || {};
  const blast = result?.mass_blast_baseline || {};
  const savings = result?.savings || {};
  const insights = result?.experiment_insights || {};

  return (
    <div className="space-y-6">
      {/* Funnel Cards */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-3">Dormancy Funnel — Who Are the Inactive Users?</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {stages.map((s, i) => (
            <div key={s} className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
              <div className="w-3 h-3 rounded-full mb-2" style={{ background: stageColors[i] }} />
              <p className="text-2xl font-bold text-white">{funnel[s] || 0}</p>
              <p className="text-xs text-slate-400 mt-1">{s}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Simulator Controls */}
      <div className="bg-slate-800/40 rounded-xl p-5 border border-slate-700/50 relative overflow-hidden backdrop-blur-sm">
        {loading && (
          <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm z-10 flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-[var(--upay-yellow)] border-t-transparent rounded-full animate-spin"></div>
          </div>
        )}
        <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
          <Zap size={16} style={{ color: 'var(--upay-yellow)' }} /> MAU Growth Simulator
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="text-xs text-slate-400">Reactivation Budget (BDT)</label>
            <input type="range" min={1000} max={100000} step={1000} value={budget}
                   onChange={e => setBudget(+e.target.value)} className="w-full mt-1 accent-[#F5A623]" />
            <span className="text-sm font-mono text-white">{budget.toLocaleString()} BDT</span>
          </div>
          <div>
            <label className="text-xs text-slate-400">Max Fatigue Cap (0-100)</label>
            <input type="range" min={0} max={100} step={5} value={fatigueCap}
                   onChange={e => setFatigueCap(+e.target.value)} className="w-full mt-1 accent-[#F5A623]" />
            <span className="text-sm font-mono text-white">{fatigueCap}</span>
          </div>
          <div>
            <label className="text-xs text-slate-400">Min Uplift Cutoff</label>
            <input type="range" min={0} max={0.5} step={0.01} value={upliftCut}
                   onChange={e => setUpliftCut(+e.target.value)} className="w-full mt-1 accent-[#F5A623]" />
            <span className="text-sm font-mono text-white">{upliftCut.toFixed(2)}</span>
          </div>
          <div>
            <label className="text-xs text-slate-400">Lifecycle Stage Filter</label>
            <select value={stage} onChange={e => setStage(e.target.value)}
                    className="w-full mt-1 bg-slate-700 text-white text-sm rounded-lg px-3 py-2 border border-slate-600">
              <option value="">All Stages</option>
              {stages.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* 4-Way Ablation Study Comparison Table */}
      {result && result.configurations && (
        <div className="bg-slate-800/40 rounded-xl p-5 border border-slate-700/50 overflow-x-auto">
          <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp size={16} /> Ablation Study: Policy Value Comparison
          </h3>
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="py-3 px-4 text-xs text-slate-400 font-medium">Metric</th>
                {result.configurations.map(c => (
                  <th key={c.name} className={`py-3 px-4 text-xs font-semibold ${
                    c.name.includes('ActivateAI') ? 'text-emerald-400' : 'text-slate-300'
                  }`}>{c.name}</th>
                ))}
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-slate-400 flex items-center gap-2"><Users size={14}/> Users Targeted</td>
                {result.configurations.map(c => <td key={c.name} className="py-3 px-4 text-white">{c.users_targeted.toLocaleString()}</td>)}
              </tr>
              <tr className="border-b border-slate-700/50 bg-slate-800/20">
                <td className="py-3 px-4 text-slate-400 flex items-center gap-2"><TrendingUp size={14}/> Incremental MAU</td>
                {result.configurations.map(c => <td key={c.name} className="py-3 px-4 font-bold text-white">{c.incremental_mau_gained.toFixed(1)}</td>)}
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-slate-400 flex items-center gap-2"><DollarSign size={14}/> Total Spend</td>
                {result.configurations.map(c => <td key={c.name} className="py-3 px-4 text-slate-300">{c.total_spend_bdt.toLocaleString()} ৳</td>)}
              </tr>
              <tr className="border-b border-slate-700/50 bg-slate-800/20">
                <td className="py-3 px-4 text-slate-400 flex items-center gap-2"><DollarSign size={14}/> Cost / MAU</td>
                {result.configurations.map(c => <td key={c.name} className="py-3 px-4 font-mono text-[var(--upay-yellow)]">{c.cost_per_incremental_mau.toLocaleString()} ৳</td>)}
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-slate-400 flex items-center gap-2"><ShieldAlert size={14}/> Cashback Waste</td>
                {result.configurations.map(c => <td key={c.name} className="py-3 px-4 text-red-400">{c.cashback_waste_bdt.toLocaleString()} ৳</td>)}
              </tr>
              <tr>
                <td className="py-3 px-4 text-slate-400 flex items-center gap-2"><ShieldAlert size={14}/> Fatigue / Opt-Out Rate</td>
                {result.configurations.map(c => <td key={c.name} className="py-3 px-4 text-orange-400">{(c.opt_out_fatigue_rate * 100).toFixed(1)}%</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Gemini Experiment Intelligence */}
      {insights.insights && (
        <div className="bg-slate-800/40 rounded-xl p-5 border border-slate-700/50">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Lightbulb size={16} style={{ color: 'var(--upay-yellow)' }} /> Experiment Intelligence
            <span className={`ml-auto text-xs px-2 py-0.5 rounded-full ${
              insights.generation_source?.includes('Gemini') ? 'bg-blue-900/50 text-blue-300' : 'bg-slate-700 text-slate-400'
            }`}>{insights.generation_source}</span>
          </h3>
          <ul className="space-y-2">
            {insights.insights.map((b, i) => (
              <li key={i} className="flex gap-2 text-sm text-slate-300">
                <span className="text-slate-500 shrink-0">{i + 1}.</span>{b}
              </li>
            ))}
          </ul>
          {insights.next_experiment && (
            <div className="mt-3 p-3 rounded-lg bg-slate-700/40 border border-slate-600/50">
              <p className="text-xs text-slate-400 mb-1">Recommended Next Experiment</p>
              <p className="text-sm text-white">{insights.next_experiment}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
