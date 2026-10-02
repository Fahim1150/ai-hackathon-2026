import React, { useEffect, useState } from 'react';
import { fetchApi } from '../api';
import { Users, Target, Frown, Moon, ShieldCheck } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function UpliftTargeting() {
  const [segments, setSegments] = useState(null);
  const [fairness, setFairness] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetchApi('/uplift/segments'),
      fetchApi('/fairness')
    ]).then(([s, f]) => {
      setSegments(s);
      setFairness(f);
      setLoading(false);
    }).catch(console.error);
  }, []);

  if (loading) return <div className="text-slate-500 animate-pulse">Loading targeting data...</div>;

  // Process fairness data for chart
  const fairnessChartData = [];
  if (fairness && fairness.region) {
    Object.entries(fairness.region.groups).forEach(([region, data]) => {
      fairnessChartData.push({
        name: region,
        'Contact Rate (%)': data.contact_rate * 100,
        'Avg Uplift (%)': data.avg_uplift * 100
      });
    });
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-800">Uplift Segments</h1>
        <p className="text-slate-500">Divide customers by how they react to offers</p>
      </header>

      {/* Segment Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-xl border-t-4 border-t-green-500 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-2 mb-2 text-green-600 font-semibold">
            <Target size={20} />
            Persuadables
          </div>
          <div className="text-3xl font-bold text-slate-800">
            {segments['Persuadable (converts because of the offer)']?.toLocaleString() || 0}
          </div>
          <div className="text-xs text-slate-500 mt-2">Converts BECAUSE of the offer. Target them!</div>
        </div>

        <div className="bg-white p-5 rounded-xl border-t-4 border-t-blue-500 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-2 mb-2 text-blue-600 font-semibold">
            <ShieldCheck size={20} />
            Sure Things
          </div>
          <div className="text-3xl font-bold text-slate-800">
            {segments['Sure Thing (converts anyway - save the budget)']?.toLocaleString() || 0}
          </div>
          <div className="text-xs text-slate-500 mt-2">Would convert anyway. Save your budget.</div>
        </div>

        <div className="bg-white p-5 rounded-xl border-t-4 border-t-slate-400 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-2 mb-2 text-slate-600 font-semibold">
            <Frown size={20} />
            Lost Causes
          </div>
          <div className="text-3xl font-bold text-slate-800">
            {segments['Lost Cause (offer does not move them)']?.toLocaleString() || 0}
          </div>
          <div className="text-xs text-slate-500 mt-2">Offer doesn't move them. Don't waste money.</div>
        </div>

        <div className="bg-white p-5 rounded-xl border-t-4 border-t-red-500 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-2 mb-2 text-red-600 font-semibold">
            <Moon size={20} />
            Sleeping Dogs
          </div>
          <div className="text-3xl font-bold text-slate-800">
            {segments['Sleeping Dog (offer may hurt - do not contact)']?.toLocaleString() || 0}
          </div>
          <div className="text-xs text-slate-500 mt-2">Offer makes them LESS likely to convert. Avoid!</div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-800 mb-4">Responsible AI: Fairness Check (by Region)</h2>
        <p className="text-sm text-slate-500 mb-6">Ensuring the model doesn't unfairly exclude demographics. The Min/Max contact rate ratio is <strong>{fairness.region.contact_rate_ratio_min_over_max}</strong> (values near 1.0 indicate demographic parity).</p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={fairnessChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
              <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
              <Tooltip cursor={{fill: '#f8fafc'}} contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
              <Legend verticalAlign="top" height={36}/>
              <Bar dataKey="Contact Rate (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} maxBarSize={40} />
              <Bar dataKey="Avg Uplift (%)" fill="#22c55e" radius={[4, 4, 0, 0]} maxBarSize={40} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
