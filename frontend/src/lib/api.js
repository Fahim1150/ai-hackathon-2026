const BASE = '/api';

export async function getOverview() {
  const r = await fetch(`${BASE}/overview`);
  return r.json();
}

export async function simulateGrowth(params) {
  const r = await fetch(`${BASE}/simulate-mau-growth`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return r.json();
}

export async function getCustomer(id) {
  const r = await fetch(`${BASE}/customer/${id}`);
  if (!r.ok) throw new Error('Customer not found');
  return r.json();
}

export async function getCustomers(params = {}) {
  const q = new URLSearchParams(params).toString();
  const r = await fetch(`${BASE}/customers?${q}`);
  return r.json();
}

export async function approveCampaign(data) {
  const r = await fetch(`${BASE}/approve-campaign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return r.json();
}
