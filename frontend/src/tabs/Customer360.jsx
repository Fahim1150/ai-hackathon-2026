import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Search, Activity, AlertTriangle, MessageSquare, Sparkles } from 'lucide-react';
import { getCustomers, getCustomer } from '../lib/api';

const Q_BG = { Persuadable: 'bg-emerald-900/50 text-emerald-300', 'Sure Thing': 'bg-blue-900/50 text-blue-300',
               'Lost Cause': 'bg-slate-700 text-slate-300', 'Sleeping Dog': 'bg-red-900/50 text-red-300' };

function FatigueGauge({ score }) {
  const pct = Math.min(score, 100);
  const color = pct < 30 ? '#10B981' : pct < 60 ? '#F5A623' : '#EF4444';
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-3 bg-slate-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-sm font-mono font-bold" style={{ color }}>{score}/100</span>
    </div>
  );
}

export default function Customer360() {
  const [customers, setCustomers] = useState([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState('');
  const [qFilter, setQFilter] = useState('');
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);

  useEffect(() => {
    const params = { page, limit: 15 };
    if (search) params.search = search;
    if (qFilter) params.quadrant = qFilter;
    getCustomers(params).then(d => { setCustomers(d.customers); setTotal(d.total); });
  }, [page, search, qFilter]);

  const selectCustomer = async (id) => {
    setLoading(true);
    try { setSelected(await getCustomer(id)); } catch (e) { console.error(e); }
    setLoading(false);
  };

  const shapData = (selected?.shap_top3 || []).map(d => ({
    feature: d.feature.replace(/_/g, ' '),
    value: d.shap_value,
    fill: d.shap_value > 0 ? '#10B981' : '#EF4444',
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6">
      {/* Customer List Panel */}
      <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 flex flex-col max-h-[calc(100vh-220px)]">
        <div className="p-3 border-b border-slate-700/50 space-y-2">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input value={search} onChange={e => { setSearch(e.target.value); setPage(1); }}
                   placeholder="Search customer ID…"
                   className="w-full bg-slate-700 text-sm text-white rounded-lg pl-9 pr-3 py-2 border border-slate-600 placeholder-slate-500" />
          </div>
          <select value={qFilter} onChange={e => { setQFilter(e.target.value); setPage(1); }}
                  className="w-full bg-slate-700 text-sm text-white rounded-lg px-3 py-1.5 border border-slate-600">
            <option value="">All Quadrants</option>
            {['Persuadable', 'Sure Thing', 'Lost Cause', 'Sleeping Dog'].map(q => <option key={q}>{q}</option>)}
          </select>
          <p className="text-xs text-slate-500">{total} users</p>
        </div>
        <div className="overflow-y-auto flex-1">
          {customers.map(c => (
            <button key={c.customer_id} onClick={() => selectCustomer(c.customer_id)}
                    className={`w-full text-left px-3 py-2.5 border-b border-slate-700/30 hover:bg-slate-700/50 transition-colors ${
                      selected?.customer_id === c.customer_id ? 'bg-slate-700/60' : ''
                    }`}>
              <div className="flex items-center justify-between">
                <span className="text-sm font-mono text-white">{c.customer_id}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${Q_BG[c.uplift_quadrant] || 'bg-slate-700 text-slate-300'}`}>
                  {c.uplift_quadrant}
                </span>
              </div>
              <div className="flex gap-3 mt-1 text-xs text-slate-500">
                <span>↑ {c.uplift_score?.toFixed(3)}</span>
                <span>{c.lifecycle_stage}</span>
              </div>
            </button>
          ))}
        </div>
        <div className="p-2 border-t border-slate-700/50 flex justify-between">
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1}
                  className="text-xs text-slate-400 hover:text-white disabled:opacity-30">← Prev</button>
          <span className="text-xs text-slate-500">Page {page}</span>
          <button onClick={() => setPage(p => p + 1)} className="text-xs text-slate-400 hover:text-white">Next →</button>
        </div>
      </div>

      {/* Detail Panel */}
      <div className="space-y-4">
        {!selected && <div className="bg-slate-800/40 rounded-xl p-10 border border-slate-700/50 text-center text-slate-500">
          <UserSearchIcon /> <p className="mt-2">Select a customer from the list</p>
        </div>}

        {loading && <p className="text-slate-400">Loading…</p>}

        {selected && !loading && (
          <>
            {/* Profile + Uplift */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
                <h4 className="text-xs text-slate-400 uppercase tracking-wide mb-3">Profile</h4>
                <div className="space-y-1.5 text-sm">
                  {[['ID', selected.customer_id], ['Stage', selected.lifecycle_stage],
                    ['Wallet', selected.wallet_type], ['Channel', selected.channel_type],
                    ['Days Inactive', selected.days_inactive], ['Tx Count', selected.historical_tx_count],
                    ['Inflow', `${selected.monthly_inflow_bdt?.toLocaleString()} ৳`],
                    ['Cash-out Ratio', (selected.cashout_ratio * 100).toFixed(1) + '%'],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-slate-400">{k}</span><span className="text-white font-mono">{v}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50 space-y-4">
                <h4 className="text-xs text-slate-400 uppercase tracking-wide">Uplift Analysis</h4>
                <div className="text-center">
                  <p className="text-4xl font-bold" style={{ color: selected.uplift_score >= 0 ? '#10B981' : '#EF4444' }}>
                    {selected.uplift_score >= 0 ? '+' : ''}{selected.uplift_score?.toFixed(4)}
                  </p>
                  <span className={`inline-block mt-1 text-xs px-2 py-0.5 rounded-full ${Q_BG[selected.uplift_quadrant]}`}>
                    {selected.uplift_quadrant}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-center text-xs">
                  <div className="bg-slate-700/40 rounded-lg p-2">
                    <p className="text-emerald-400 font-bold">{selected.prob_active_treated?.toFixed(3)}</p>
                    <p className="text-slate-500">P(active|treated)</p>
                  </div>
                  <div className="bg-slate-700/40 rounded-lg p-2">
                    <p className="text-blue-400 font-bold">{selected.prob_active_control?.toFixed(3)}</p>
                    <p className="text-slate-500">P(active|control)</p>
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <AlertTriangle size={14} className="text-slate-400" />
                    <span className="text-xs text-slate-400">Offer Fatigue</span>
                  </div>
                  <FatigueGauge score={selected.offer_fatigue_score} />
                </div>
              </div>
            </div>

            {/* SHAP Chart */}
            <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
              <h4 className="text-xs text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
                <Activity size={14} /> Top 3 SHAP Drivers
              </h4>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={shapData} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" tick={{ fill: '#94A3B8', fontSize: 12 }} />
                  <YAxis dataKey="feature" type="category" tick={{ fill: '#94A3B8', fontSize: 12 }} width={130} />
                  <Tooltip contentStyle={{ background: '#1E293B', border: '1px solid #334155', borderRadius: 8 }} />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]} name="SHAP Value">
                    {shapData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Offer + SMS Preview */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
                <h4 className="text-xs text-slate-400 uppercase tracking-wide mb-2">Recommended Offer</h4>
                <p className="text-sm font-semibold" style={{ color: 'var(--upay-yellow)' }}>{selected.assigned_offer}</p>
                <p className="text-xs text-slate-500 mt-1">Cost: {selected.assigned_offer_cost_bdt} BDT</p>
                <p className="text-xs text-slate-500">Affinity: {selected.top_affinity_domain}</p>
              </div>

              <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare size={14} className="text-slate-400" />
                  <h4 className="text-xs text-slate-400 uppercase tracking-wide">SMS Preview</h4>
                  <span className={`ml-auto text-[10px] px-1.5 py-0.5 rounded-full ${
                    selected.nudge?.generation_source?.includes('Gemini') ? 'bg-blue-900/50 text-blue-300' : 'bg-slate-700 text-slate-400'
                  }`}>
                    <Sparkles size={10} className="inline mr-1" />{selected.nudge?.generation_source}
                  </span>
                </div>
                <div className="space-y-2">
                  <div className="bg-slate-700/40 rounded-lg p-2.5">
                    <p className="text-[10px] text-slate-500 mb-0.5">English</p>
                    <p className="text-xs text-white">{selected.nudge?.sms_english}</p>
                  </div>
                  <div className="bg-slate-700/40 rounded-lg p-2.5">
                    <p className="text-[10px] text-slate-500 mb-0.5">বাংলা</p>
                    <p className="text-xs text-white">{selected.nudge?.sms_bangla}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Marketer Explanation */}
            {selected.nudge?.marketer_explanation && (
              <div className="bg-slate-800/40 rounded-xl p-4 border border-slate-700/50">
                <h4 className="text-xs text-slate-400 uppercase tracking-wide mb-2 flex items-center gap-2">
                  <Sparkles size={14} /> Marketer Explanation
                </h4>
                <p className="text-sm text-slate-300">{selected.nudge.marketer_explanation}</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function UserSearchIcon() {
  return <Search size={32} className="mx-auto text-slate-600" />;
}
