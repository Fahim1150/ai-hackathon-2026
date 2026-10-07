const BASE = '/api';

// Dev API key for local development — in production this would come from
// a login flow / session token / environment variable injected at build time.
const API_KEY = 'upay-activate-ai-dev-key-2026';
const AUTH_HEADERS = {
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
};

export async function getOverview() {
  const r = await fetch(`${BASE}/overview`);
  return r.json();
}

export async function simulateGrowth(params) {
  const r = await fetch(`${BASE}/simulate-mau-growth`, {
    method: 'POST',
    headers: AUTH_HEADERS,
    body: JSON.stringify(params),
  });
  return r.json();
}

export async function getCustomer(id) {
  const r = await fetch(`${BASE}/customer/${id}`, {
    headers: { 'X-API-Key': API_KEY },
  });
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
    headers: AUTH_HEADERS,
    body: JSON.stringify(data),
  });
  return r.json();
}

export async function getApprovals() {
  const r = await fetch(`${BASE}/approvals`, {
    headers: { 'X-API-Key': API_KEY },
  });
  return r.json();
}
