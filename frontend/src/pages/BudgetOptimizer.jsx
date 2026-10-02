import React, { useState } from 'react';
import { fetchApi } from '../api';
import { Calculator, CheckCircle2, ChevronRight, BarChart3, PieChart as PieChartIcon } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as PieTooltip, Legend as PieLegend, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTooltip } from 'recharts';

const COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899'];

export default function BudgetOptimizer() {
  const [budget, setBudget] = useState(20000);
  const [maxCost, setMaxCost] = useState(30);
  const [loading, setLoading] = useState(false);
  const [plan, setPlan] = useState(null);

  const handleOptimize = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetchApi('/budget/optimize', {
        method: 'POST',
        body: JSON.stringify({
          budget_bdt: Number(budget),
          max_per_user_cost: Number(maxCost)
        })
      });
      setPlan(res);
    } catch (err) {
      console.error(err);
      alert("Failed to optimize budget");
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async () => {
    if (!plan || !plan.campaign_id) return;
    try {
      const res = await fetchApi(`/campaigns/${plan.campaign_id}/approve`, { method: 'POST' });
      setPlan({ ...plan, status: res.status });
    } catch (err) {
      alert("Approval failed");
    }
  };

  // Prepare chart data
  let pieData = [];
  if (plan && plan.offers_assigned) {
    pieData = Object.entries(plan.offers_assigned).map(([name, value]) => ({ name, value }));
  }

  const comparisonData = plan ? [
    {
      name: 'Baseline (Equal Split)',
      'Gain (BDT)': plan.baseline_comparison.gain_vs_baseline_bdt,
    },
    {
      name: 'CampaignIQ Uplift Model',
      'Gain (BDT)': plan.expected_incremental_profit_bdt,
    }
  ] : [];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-800">Budget Optimizer</h1>
        <p className="text-slate-500">Allocate budget to maximize incremental conversions</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Constraints Form */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm lg:col-span-1 h-fit">
          <h2 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <Calculator size={20} />
            Campaign Constraints
          </h2>
          <form onSubmit={handleOptimize} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Total Budget (BDT)</label>
              <input
                type="number"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                required
                min="1000"
              />
              <div className="mt-2 flex items-center gap-2">
                <input 
                  type="range" 
                  min="5000" max="100000" step="1000" 
                  value={budget} 
                  onChange={(e) => setBudget(e.target.value)}
                  className="w-full accent-green-500"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Max Cost per User (BDT)</label>
              <input
                type="number"
                value={maxCost}
                onChange={(e) => setMaxCost(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                required
                min="5"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-slate-900 text-white font-medium py-2.5 rounded-lg hover:bg-slate-800 transition-colors flex justify-center items-center gap-2 disabled:opacity-50"
            >
              {loading ? 'Optimizing...' : 'Run Optimizer'}
              <ChevronRight size={18} />
            </button>
          </form>
        </div>

        {/* Results Panel */}
        {plan && (
          <div className="lg:col-span-2 space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <div className="text-xs text-slate-500 mb-1">Users Selected</div>
                <div className="text-xl font-bold text-slate-800">{plan.selected_users.toLocaleString()}</div>
                <div className="text-xs text-slate-400">of {plan.eligible_users.toLocaleString()} eligible</div>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <div className="text-xs text-slate-500 mb-1">Expected Spend</div>
                <div className="text-xl font-bold text-slate-800">{plan.expected_spend_bdt.toLocaleString()} ৳</div>
                <div className="text-xs text-slate-400">Under {budget} ৳ budget</div>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm border-l-4 border-l-green-500">
                <div className="text-xs text-slate-500 mb-1">Expected Incremental Profit</div>
                <div className="text-xl font-bold text-green-600">+{plan.expected_incremental_profit_bdt.toLocaleString()} ৳</div>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <div className="text-xs text-slate-500 mb-1">Campaign ROI</div>
                <div className="text-xl font-bold text-slate-800">{plan.roi_percentage}%</div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h3 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
                  <PieChartIcon size={16} /> Offer Allocation
                </h3>
                <div className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={pieData} innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <PieTooltip formatter={(value) => value.toLocaleString()} />
                      <PieLegend verticalAlign="bottom" height={36}/>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h3 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
                  <BarChart3 size={16} /> Gain vs Baseline
                </h3>
                <div className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                      <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12 }} />
                      <BarTooltip cursor={{fill: '#f8fafc'}} />
                      <Bar dataKey="Gain (BDT)" radius={[4, 4, 0, 0]}>
                        {comparisonData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={index === 0 ? '#94a3b8' : '#22c55e'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="bg-slate-900 p-6 rounded-xl shadow-md flex items-center justify-between text-white">
              <div>
                <div className="font-semibold text-lg">Campaign Status</div>
                <div className="text-sm text-slate-300">
                  {plan.status === 'APPROVED' 
                    ? 'Campaign is ready for launch.' 
                    : 'Human approval is required before launch.'}
                </div>
              </div>
              
              {plan.status === 'APPROVED' ? (
                <div className="flex items-center gap-2 text-green-400 font-medium px-4 py-2 bg-green-400/10 rounded-lg">
                  <CheckCircle2 size={20} />
                  Approved
                </div>
              ) : (
                <button 
                  onClick={handleApprove}
                  className="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                >
                  Approve Campaign
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
