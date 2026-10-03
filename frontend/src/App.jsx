import { useState } from 'react';
import { BarChart3, Users, UserSearch, ShieldCheck } from 'lucide-react';
import DormancyFunnel from './tabs/DormancyFunnel';
import UpliftExplorer from './tabs/UpliftExplorer';
import Customer360 from './tabs/Customer360';
import ResponsibleAI from './tabs/ResponsibleAI';

const TABS = [
  { id: 'funnel', label: 'Dormancy Funnel & MAU Simulator', icon: BarChart3 },
  { id: 'uplift', label: 'Solo-Utility Habit & Uplift', icon: Users },
  { id: 'customer', label: 'Customer 360 & SHAP', icon: UserSearch },
  { id: 'responsible', label: 'Responsible AI & Oversight', icon: ShieldCheck },
];

export default function App() {
  const [tab, setTab] = useState('funnel');

  return (
    <div className="min-h-screen" style={{ background: 'var(--upay-dark)' }}>
      {/* Header */}
      <header className="border-b border-slate-700/50 px-6 py-4"
              style={{ background: 'var(--upay-navy)' }}>
        <div className="max-w-[1400px] mx-auto flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center font-bold text-lg"
                 style={{ background: 'var(--upay-yellow)', color: 'var(--upay-navy)' }}>
              u
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">upay ActivateAI</h1>
              <p className="text-xs text-slate-400">Dormant-to-Active Lifecycle &amp; Uplift Engine</p>
            </div>
          </div>
          <span className="ml-auto text-xs px-2 py-1 rounded-full border border-slate-600 text-slate-400">
            Track 04 · Growth &amp; Campaign Intelligence
          </span>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="border-b border-slate-700/50 px-6"
           style={{ background: 'var(--upay-navy)' }}>
        <div className="max-w-[1400px] mx-auto flex gap-1 overflow-x-auto">
          {TABS.map(t => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  active
                    ? 'text-white border-[var(--upay-yellow)]'
                    : 'text-slate-400 border-transparent hover:text-slate-200 hover:border-slate-500'
                }`}
              >
                <Icon size={16} />
                {t.label}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Tab Content */}
      <main className="max-w-[1400px] mx-auto p-6">
        {tab === 'funnel' && <DormancyFunnel />}
        {tab === 'uplift' && <UpliftExplorer />}
        {tab === 'customer' && <Customer360 />}
        {tab === 'responsible' && <ResponsibleAI />}
      </main>
    </div>
  );
}
