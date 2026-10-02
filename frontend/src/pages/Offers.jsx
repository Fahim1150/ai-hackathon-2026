import React, { useState } from 'react';
import { fetchApi } from '../api';
import { Search, Info, Check, X, ShieldAlert, Award, TrendingDown } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Offers() {
  const [userId, setUserId] = useState('SYN_100042');
  const [data, setData] = useState(null);
  const [fatigue, setFatigue] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!userId) return;
    
    setLoading(true);
    setError('');
    try {
      const [offerRes, fatigueRes] = await Promise.all([
        fetchApi(`/customers/${userId}/offers`),
        fetchApi('/campaigns/fatigue')
      ]);
      setData(offerRes);
      setFatigue(fatigueRes);
    } catch (err) {
      setError(err.message || "Customer not found");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const fatigueChartData = fatigue ? fatigue.response_rate_trend.map((rate, i) => ({
    send: `Send ${i+1}`,
    'Response Rate': rate * 100
  })) : [];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-800">Customer Offers</h1>
        <p className="text-slate-500">Next-best-offer ranking with SHAP explainability</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Search */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <form onSubmit={handleSearch} className="flex gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-2.5 text-slate-400" size={20} />
                <input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="Enter Customer ID (e.g. SYN_100042)"
                  className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                />
              </div>
              <button 
                type="submit"
                disabled={loading}
                className="bg-slate-900 text-white px-6 py-2 rounded-lg font-medium hover:bg-slate-800 disabled:opacity-50"
              >
                {loading ? 'Searching...' : 'Lookup'}
              </button>
            </form>
            {error && <div className="mt-3 text-red-500 text-sm">{error}</div>}
          </div>

          {/* Results */}
          {data && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="bg-slate-50 p-4 border-b border-slate-200 flex justify-between items-center">
                <div>
                  <h2 className="font-semibold text-slate-800">Customer: {data.user_id}</h2>
                  <div className="text-sm text-slate-500 mt-1">Segment: <span className="font-medium text-slate-700">{data.segment}</span></div>
                </div>
                
                {data.allowed_to_contact ? (
                  <div className="flex items-center gap-1.5 px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                    <Check size={16} /> Contact Allowed
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm font-medium">
                    <X size={16} /> Blocked by Rules
                  </div>
                )}
              </div>
              
              {!data.allowed_to_contact && data.blocking_rules && (
                <div className="p-4 bg-red-50 border-b border-red-100 text-sm text-red-800 flex gap-2">
                  <ShieldAlert size={18} className="shrink-0 text-red-500" />
                  <div>
                    <strong>Blocked because:</strong> {data.blocking_rules.join('; ')}
                  </div>
                </div>
              )}

              <div className="divide-y divide-slate-100">
                {data.ranked_offers.map((offer, idx) => (
                  <div key={offer.offer_id} className={`p-5 ${idx === 0 ? 'bg-green-50/30' : ''}`}>
                    <div className="flex justify-between items-start mb-4">
                      <div className="flex items-center gap-2">
                        {idx === 0 && <Award className="text-green-500" size={24} />}
                        <h3 className="font-bold text-slate-800 text-lg">{offer.offer_name}</h3>
                      </div>
                      <div className="text-right">
                        <div className="text-green-600 font-bold text-lg">+{offer.expected_gain.toFixed(2)} ৳</div>
                        <div className="text-xs text-slate-400">Expected Gain</div>
                      </div>
                    </div>
                    
                    <div className="flex gap-6 mb-4 text-sm">
                      <div>
                        <span className="text-slate-500 block text-xs">Uplift</span>
                        <span className="font-medium text-slate-800">{(offer.uplift * 100).toFixed(2)}%</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-xs">Prob (Treated)</span>
                        <span className="font-medium text-slate-800">{(offer.prob_treated * 100).toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-xs">Offer Cost</span>
                        <span className="font-medium text-slate-800">{offer.cost.toFixed(1)} ৳</span>
                      </div>
                    </div>

                    <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
                      <div className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider flex items-center gap-1">
                        <Info size={14} /> Top drivers (SHAP)
                      </div>
                      <ul className="space-y-1.5">
                        {offer.reasons.map((r, i) => (
                          <li key={i} className="text-sm flex items-start gap-2">
                            <span className={r.effect.includes('raises') ? 'text-green-500' : 'text-red-500'}>
                              {r.effect.includes('raises') ? '↑' : '↓'}
                            </span>
                            <span className="text-slate-700">{r.reason}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Fatigue Sidebar */}
        {fatigue && (
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm h-fit">
            <h2 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <TrendingDown size={20} className="text-amber-500" />
              Fatigue Monitor
            </h2>
            <div className="mb-4">
              <div className="text-3xl font-bold text-slate-800">{(fatigue.fatigue_percentage * 100).toFixed(1)}%</div>
              <div className="text-sm text-slate-500">of global audience fatigued</div>
            </div>
            
            <div className="h-40 mb-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={fatigueChartData} margin={{top: 5, right: 5, left: -20, bottom: 0}}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="send" tick={{fontSize: 10}} />
                  <YAxis tick={{fontSize: 10}} />
                  <Tooltip />
                  <Line type="monotone" dataKey="Response Rate" stroke="#f59e0b" strokeWidth={2} dot={{r: 3}} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="text-sm text-slate-600 bg-amber-50 p-3 rounded-lg border border-amber-100">
              Model detects diminishing returns on repeated sends. Rules engine currently blocks offers for users with 3+ contacts in 30 days.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
