import { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { getOverview } from '../lib/api';

const Q_COLORS = { Persuadable: '#10B981', 'Sure Thing': '#3B82F6', 'Lost Cause': '#6B7280', 'Sleeping Dog': '#EF4444' };
const OFFER_COLORS = ['#F5A623', '#3B82F6', '#10B981', '#8B5CF6'];

export default function UpliftExplorer() {
  const [data, setData] = useState(null);
  useEffect(() => { getOverview().then(setData).catch(console.error); }, []);
  if (!data) return <p className="text-slate-400">Loading…</p>;

  const quadDist = Object.entries(data.uplift_quadrant_distribution || {}).map(([name, value]) => ({ name, value }));
  const offerDist = Object.entries(data.offer_distribution || {}).map(([name, value]) => ({ name: name.replace(/^(Zero-Fee |10% )/, '').slice(0, 25), value, full: name }));

  const funnelByStage = Object.entries(data.fairness?.lifecycle_stage?.groups || {}).map(([name, v]) => ({
    name: name.replace(' ', '\n'), mean_uplift: v.mean_uplift, count: v.count,
  }));

  const metrics = data.validation_metrics || {};

  return (
    <div className="space-y-6">
      {/* Validation Metrics Bar */}
      <div className="bg-slate-800/40 rounded-xl p-4 border border-slate-700/50">
        <h3 className="text-sm font-semibold text-white mb-4 text-center">T-Learner Validation (95% CI via Bootstrap)</h3>
        <div className="flex flex-wrap gap-6 justify-center">
          {metrics.t_learner && Object.entries(metrics.t_learner).map(([k, v]) => (
            <div key={k} className="text-center">
              <p className="text-xl font-bold" style={{ color: 'var(--upay-yellow)' }}>
                {v.mean.toFixed(4)}
              </p>
              <p className="text-[10px] text-slate-500 mb-1">[{v.ci_lower.toFixed(3)} - {v.ci_upper.toFixed(3)}]</p>
              <p className="text-xs text-slate-400">{k.replace(/_/g, ' ').toUpperCase()}</p>
            </div>
          ))}
        </div>
      </div>

      {/* OPE Metrics Bar */}
      {data.ope_doubly_robust && (
        <div className="bg-emerald-950/20 rounded-xl p-4 border border-emerald-900/30">
          <h3 className="text-sm font-semibold text-emerald-400 mb-4 text-center">Doubly Robust Off-Policy Evaluation</h3>
          <div className="flex flex-wrap gap-6 justify-center">
            {Object.entries(data.ope_doubly_robust).map(([k, v]) => (
              <div key={k} className="text-center">
                <p className="text-lg font-bold text-emerald-300">{(v * 100).toFixed(2)}%</p>
                <p className="text-xs text-slate-400">{k.replace(/_/g, ' ').toUpperCase()}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quadrant Pie */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700/50">
          <h3 className="text-sm font-semibold text-white mb-4">Uplift Quadrant Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={quadDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                   outerRadius={100} innerRadius={50} paddingAngle={3} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {quadDist.map(d => <Cell key={d.name} fill={Q_COLORS[d.name] || '#666'} />)}
              </Pie>
              <Tooltip contentStyle={{ background: '#1E293B', border: '1px solid #334155', borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Offer Distribution */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700/50">
          <h3 className="text-sm font-semibold text-white mb-4">Solo-Utility Offer Allocation</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={offerDist} layout="vertical" margin={{ left: 10 }}>
              <XAxis type="number" tick={{ fill: '#94A3B8', fontSize: 12 }} />
              <YAxis dataKey="name" type="category" tick={{ fill: '#94A3B8', fontSize: 11 }} width={140} />
              <Tooltip contentStyle={{ background: '#1E293B', border: '1px solid #334155', borderRadius: 8 }}
                       formatter={(v, _, p) => [v, p.payload.full]} />
              <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                {offerDist.map((_, i) => <Cell key={i} fill={OFFER_COLORS[i % OFFER_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Uplift by Lifecycle Stage */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700/50 lg:col-span-2">
          <h3 className="text-sm font-semibold text-white mb-4">Mean Uplift by Lifecycle Stage</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={funnelByStage} margin={{ left: 10 }}>
              <XAxis dataKey="name" tick={{ fill: '#94A3B8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94A3B8', fontSize: 12 }} />
              <Tooltip contentStyle={{ background: '#1E293B', border: '1px solid #334155', borderRadius: 8 }} />
              <Legend />
              <Bar dataKey="mean_uplift" fill="#F5A623" name="Mean Uplift" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
