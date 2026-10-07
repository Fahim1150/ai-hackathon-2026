import { useState, useEffect } from 'react';
import { ShieldCheck, CheckCircle, AlertCircle, User } from 'lucide-react';
import { getOverview, approveCampaign } from '../lib/api';
import toast from 'react-hot-toast';

const AI_CHECKS = [
  { cat: 'Privacy', items: ['All data is 100% synthetic — no real PII', 'No real phone numbers or account numbers', 'Gemini API receives only structured summaries'] },
  { cat: 'Explainability', items: ['Every recommendation traced to top 3 SHAP drivers', 'SHAP values pre-computed and stored', 'generation_source field discloses AI vs fallback'] },
  { cat: 'Fairness', items: ['Contact rates audited across lifecycle_stage and wallet_type', 'Min/max ratio monitored (>0.7 target)', 'Remittance & USSD users explicitly tracked'] },
  { cat: 'Security', items: ['No API keys committed to repository', 'GEMINI_API_KEY optional — system works without it', 'Official google-genai SDK with structured outputs'] },
  { cat: 'Human Oversight', items: ['No campaign dispatched without reviewer approval', 'Approval logs: reviewer, timestamp, budget, notes', 'Budget optimizer respects hard caps'] },
];

export default function ResponsibleAI() {
  const [overview, setOverview] = useState(null);
  const [name, setName] = useState('');
  const [budgetInput, setBudgetInput] = useState('10000');
  const [notes, setNotes] = useState('');
  const [approvals, setApprovals] = useState([]);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => { getOverview().then(setOverview).catch(console.error); }, []);

  const handleApprove = async () => {
    if (!name.trim()) {
      toast.error('Approver Name is required');
      return;
    }
    setSubmitting(true);
    const toastId = toast.loading('Authorizing campaign...');
    try {
      const r = await approveCampaign({ admin_user: name, allocated_budget: +budgetInput, notes });
      setApprovals(prev => [r, ...prev]);
      setName(''); setNotes('');
      toast.success(`Campaign approved by ${name}!`, { id: toastId });
    } catch (e) {
      console.error(e);
      toast.error('Failed to approve campaign', { id: toastId });
    }
    setSubmitting(false);
  };

  const fairness = overview?.fairness || {};

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Responsible AI Checklist */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700/50">
          <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <ShieldCheck size={16} style={{ color: 'var(--upay-yellow)' }} /> Responsible AI Checklist
          </h3>
          <div className="space-y-4">
            {AI_CHECKS.map(c => (
              <div key={c.cat}>
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1.5">{c.cat}</p>
                {c.items.map((item, i) => (
                  <div key={i} className="flex items-start gap-2 py-0.5">
                    <CheckCircle size={14} className="text-emerald-500 shrink-0 mt-0.5" />
                    <span className="text-xs text-slate-300">{item}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Fairness Audit */}
        <div className="space-y-4">
          {/* Wallet Fairness */}
          <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700/50">
            <h3 className="text-sm font-semibold text-white mb-3">Fairness Audit — Wallet Type</h3>
            {fairness.wallet_type && (
              <>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xs text-slate-400">Min/Max Persuadable Ratio:</span>
                  <span className={`text-sm font-bold ${
                    fairness.wallet_type.min_max_persuadable_ratio >= 0.7 ? 'text-emerald-400' : 'text-amber-400'
                  }`}>{fairness.wallet_type.min_max_persuadable_ratio}</span>
                  {fairness.wallet_type.min_max_persuadable_ratio < 0.7 && (
                    <AlertCircle size={14} className="text-amber-400" />
                  )}
                </div>
                <div className="space-y-2">
                  {Object.entries(fairness.wallet_type.groups || {}).map(([w, v]) => (
                    <div key={w} className="bg-slate-700/40 rounded-lg p-3 flex items-center justify-between">
                      <div>
                        <p className="text-sm text-white font-medium">{w}</p>
                        <p className="text-xs text-slate-500">{v.count} users</p>
                      </div>
                      <div className="text-right text-xs">
                        <p className="text-slate-300">Uplift: <span className="font-mono">{v.mean_uplift}</span></p>
                        <p className="text-slate-300">Contact: <span className="font-mono">{(v.persuadable_rate * 100).toFixed(1)}%</span></p>
                        <p className="text-slate-400">Fatigue: <span className="font-mono">{v.mean_fatigue}</span></p>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Lifecycle Fairness */}
          <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700/50">
            <h3 className="text-sm font-semibold text-white mb-3">Fairness Audit — Lifecycle Stage</h3>
            {fairness.lifecycle_stage && (
              <>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xs text-slate-400">Min/Max Persuadable Ratio:</span>
                  <span className={`text-sm font-bold ${
                    fairness.lifecycle_stage.min_max_persuadable_ratio >= 0.7 ? 'text-emerald-400' : 'text-amber-400'
                  }`}>{fairness.lifecycle_stage.min_max_persuadable_ratio}</span>
                  {fairness.lifecycle_stage.min_max_persuadable_ratio < 0.7 && (
                    <AlertCircle size={14} className="text-amber-400" />
                  )}
                </div>
                <div className="space-y-2">
                  {Object.entries(fairness.lifecycle_stage.groups || {}).map(([s, v]) => (
                    <div key={s} className="bg-slate-700/40 rounded-lg p-3 flex items-center justify-between">
                      <div>
                        <p className="text-sm text-white font-medium">{s}</p>
                        <p className="text-xs text-slate-500">{v.count} users</p>
                      </div>
                      <div className="text-right text-xs">
                        <p className="text-slate-300">Uplift: <span className="font-mono">{v.mean_uplift}</span></p>
                        <p className="text-slate-300">Contact: <span className="font-mono">{(v.persuadable_rate * 100).toFixed(1)}%</span></p>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Human Approval */}
      <div className="bg-slate-800/60 rounded-xl p-5 border-2 border-amber-900/50">
        <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
          <User size={16} style={{ color: 'var(--upay-yellow)' }} /> Human Reviewer Campaign Approval
          <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-amber-900/40 text-amber-300">
            PENDING HUMAN APPROVAL
          </span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <div>
            <label className="text-xs text-slate-400">Reviewer Name *</label>
            <input value={name} onChange={e => setName(e.target.value)}
                   className="w-full bg-slate-700 text-white text-sm rounded-lg px-3 py-2 mt-1 border border-slate-600" />
          </div>
          <div>
            <label className="text-xs text-slate-400">Approved Budget (BDT)</label>
            <input type="number" value={budgetInput} onChange={e => setBudgetInput(e.target.value)}
                   className="w-full bg-slate-700 text-white text-sm rounded-lg px-3 py-2 mt-1 border border-slate-600" />
          </div>
          <div>
            <label className="text-xs text-slate-400">Notes</label>
            <input value={notes} onChange={e => setNotes(e.target.value)}
                   className="w-full bg-slate-700 text-white text-sm rounded-lg px-3 py-2 mt-1 border border-slate-600" />
          </div>
        </div>
        <button onClick={handleApprove} disabled={!name.trim() || submitting}
                className="px-6 py-2.5 rounded-lg text-sm font-semibold transition-colors disabled:opacity-40"
                style={{ background: 'var(--upay-yellow)', color: 'var(--upay-navy)' }}>
          {submitting ? 'Submitting…' : '✅ Approve & Authorize Campaign Launch'}
        </button>

        {/* Approval Log */}
        {approvals.length > 0 && (
          <div className="mt-4 space-y-2">
            <p className="text-xs text-slate-400 uppercase tracking-wide">Approval Log</p>
            {approvals.map((a, i) => (
              <div key={i} className="bg-emerald-950/30 border border-emerald-900/40 rounded-lg p-3 flex items-center gap-4">
                <CheckCircle size={16} className="text-emerald-400 shrink-0" />
                <div className="text-xs text-emerald-300">
                  <span className="font-semibold">{a.campaign_id}</span> approved by <span className="font-semibold">{a.admin_user}</span>
                  {' '} — {a.allocated_budget.toLocaleString()} BDT — {new Date(a.timestamp).toLocaleString()}
                  {a.notes && <span className="text-emerald-400/60"> — "{a.notes}"</span>}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Synthetic Data Note */}
      <div className="bg-slate-800/40 rounded-xl p-4 border border-slate-700/50">
        <h4 className="text-xs text-slate-400 uppercase tracking-wide mb-2">Synthetic Data Disclaimer</h4>
        <p className="text-xs text-slate-500">
          This prototype uses 100% synthetic data generated by <code className="text-slate-400">backend/data_generator.py</code> with
          seed=42. No real upay user data, PII, or transaction records are used at any stage.
          Ground-truth causal patterns are planted to demonstrate uplift modeling capabilities.
          See <code className="text-slate-400">docs/LOGIC_CHAIN.md</code> §3 for full assumptions.
        </p>
      </div>
    </div>
  );
}
