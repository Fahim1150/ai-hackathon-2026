import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, Calculator, Gift, AlertTriangle } from 'lucide-react';

import Dashboard from './pages/Dashboard';
import UpliftTargeting from './pages/UpliftTargeting';
import BudgetOptimizer from './pages/BudgetOptimizer';
import Offers from './pages/Offers';

function Layout({ children }) {
  return (
    <div className="flex h-screen bg-slate-50 font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-slate-900 text-white flex flex-col">
        <div className="p-6 border-b border-slate-800">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <span className="text-green-500">Campaign</span>IQ
          </h1>
          <p className="text-xs text-slate-400 mt-1">Uplift & Next-Best-Offer</p>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          <NavLink to="/" className={({isActive}) => `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive ? 'bg-green-500/10 text-green-400' : 'hover:bg-slate-800'}`}>
            <LayoutDashboard size={20} />
            <span>Dashboard</span>
          </NavLink>
          <NavLink to="/targeting" className={({isActive}) => `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive ? 'bg-green-500/10 text-green-400' : 'hover:bg-slate-800'}`}>
            <Users size={20} />
            <span>Uplift Targeting</span>
          </NavLink>
          <NavLink to="/budget" className={({isActive}) => `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive ? 'bg-green-500/10 text-green-400' : 'hover:bg-slate-800'}`}>
            <Calculator size={20} />
            <span>Budget Optimizer</span>
          </NavLink>
          <NavLink to="/offers" className={({isActive}) => `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive ? 'bg-green-500/10 text-green-400' : 'hover:bg-slate-800'}`}>
            <Gift size={20} />
            <span>Customer Offers</span>
          </NavLink>
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Banner */}
        <div className="bg-amber-100 border-b border-amber-200 px-6 py-2 flex items-center justify-center gap-2 text-amber-800 text-sm font-medium shadow-sm z-10">
          <AlertTriangle size={16} />
          <span>⚠️ Synthetic data — not real customer information</span>
        </div>
        
        {/* Page Content */}
        <main className="flex-1 overflow-y-auto p-8">
          {children}
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/targeting" element={<UpliftTargeting />} />
          <Route path="/budget" element={<BudgetOptimizer />} />
          <Route path="/offers" element={<Offers />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
