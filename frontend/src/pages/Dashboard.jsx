import React, { useEffect, useState } from 'react';
import { fetchApi } from '../api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, AlertCircle, Zap, Activity } from 'lucide-react';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [fatigue, setFatigue] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetchApi('/metrics'),
      fetchApi('/campaigns/fatigue')
    ]).then(([m, f]) => {
      setMetrics(m);
      setFatigue(f);
      setLoading(false);
    }).catch(console.error);
  }, []);

  if (loading) return <div className="text-slate-500 animate-pulse">Loading dashboard...</div>;

  // Prepare chart data
  const chartData = metrics ? Object.entries(metrics.policies).map(([name, data]) => ({
    name: name.replace('Blast ', '').replace(' to everyone', ''),
    'Incremental Profit (BDT)': data.incremental_vs_no_campaign,
    'Contact Rate (%)': data.contact_rate * 100
  })) : [];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-800">Overview</h1>
        <p className="text-slate-500">Campaign intelligence and uplift metrics</p>
      </header>

      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-sm font-medium text-slate-500 flex items-center gap-2">
            <Zap size={16} className="text-amber-500" />
            Max Incremental Profit
          </div>
          <div className="mt-2 text-3xl font-bold text-slate-800">
            +{metrics.policies['Uplift model + business rules'].incremental_vs_no_campaign.toFixed(2)} ৳
          </div>
          <div className="mt-1 text-xs text-slate-400">per targeted user vs baseline</div>
        </div>
        
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-sm font-medium text-slate-500 flex items-center gap-2">
            <Activity size={16} className="text-blue-500" />
            Total Train/Test Users
          </div>
          <div className="mt-2 text-3xl font-bold text-slate-800">
            {(metrics.train_rows + metrics.test_rows).toLocaleString()}
          </div>
          <div className="mt-1 text-xs text-slate-400">Synthetic population</div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-sm font-medium text-slate-500 flex items-center gap-2">
            <TrendingUp size={16} className="text-green-500" />
            Active Offers
          </div>
          <div className="mt-2 text-3xl font-bold text-slate-800">
            {metrics.n_offers}
          </div>
          <div className="mt-1 text-xs text-slate-400">Unique treatment arms</div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-sm font-medium text-slate-500 flex items-center gap-2">
            <AlertCircle size={16} className="text-red-500" />
            Fatigued Users
          </div>
          <div className="mt-2 text-3xl font-bold text-slate-800">
            {(fatigue.fatigue_percentage * 100).toFixed(1)}%
          </div>
          <div className="mt-1 text-xs text-slate-400">Received 3+ offers in 30d</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Chart */}
        <div className="lg:col-span-2 bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">Policy Performance (Expected Incremental Profit)</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#64748b' }} interval={0} angle={-15} textAnchor="end" />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
                <Tooltip cursor={{fill: '#f8fafc'}} contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                <Legend verticalAlign="top" height={36}/>
                <Bar dataKey="Incremental Profit (BDT)" fill="#22c55e" radius={[4, 4, 0, 0]} maxBarSize={50} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* AI Recommendations & Alerts */}
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">AI Recommendations</h2>
            <div className="space-y-3">
              <div className="p-3 bg-green-50 rounded-lg border border-green-100 flex gap-3">
                <Zap className="text-green-600 shrink-0" size={20} />
                <div>
                  <div className="font-medium text-green-900 text-sm">Scale Uplift Targeting</div>
                  <div className="text-xs text-green-700 mt-1">Uplift model + business rules generates +{metrics.policies['Uplift model + business rules'].incremental_vs_no_campaign.toFixed(2)} BDT per user vs equal-split.</div>
                </div>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg border border-blue-100 flex gap-3">
                <TrendingUp className="text-blue-600 shrink-0" size={20} />
                <div>
                  <div className="font-medium text-blue-900 text-sm">Test "Referral Bonus" limits</div>
                  <div className="text-xs text-blue-700 mt-1">Shows high Qini AUC. Consider running an A/B test with higher rewards.</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">Fatigue Alerts</h2>
            {fatigue.alerts && fatigue.alerts.length > 0 ? (
              <div className="space-y-3">
                {fatigue.alerts.map((alert, i) => (
                  <div key={i} className="p-3 bg-red-50 rounded-lg border border-red-100 flex gap-3">
                    <AlertCircle className="text-red-600 shrink-0" size={20} />
                    <div>
                      <div className="font-medium text-red-900 text-sm">{alert.message}</div>
                      <div className="text-xs text-red-700 mt-1">Suggestion: {alert.suggestion}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-slate-500">No active fatigue alerts. Audience is healthy.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
